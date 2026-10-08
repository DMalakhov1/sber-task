"""Optional local reranking. No credentials or paid API calls; keep provenance intact."""
from collections import Counter
import numpy as np
from task2_bot.rag.search import source_scope

DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class LocalReranker:
    def __init__(self, model_name=DEFAULT_MODEL, device="cpu", model=None):
        self.model_name, self.device, self.model = model_name, device, model

    def score(self, question, hits, alternate_query=None):
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device,
                                      max_length=512, trust_remote_code=False)
        # English translation for English reports; retain Russian for the Russian report.
        pairs = [(alternate_query if alternate_query and h['source_id'].startswith('iea_')
                  else question, h['title'] + "\n" + h['text']) for h in hits]
        return self.model.predict(pairs, batch_size=8, show_progress_bar=False,
                                  convert_to_numpy=True)


class RerankingRetriever:
    """Call under the service search lock. Failed model stays disabled until restart."""
    def __init__(self, retriever, reranker=None, candidates=24, strict=False):
        if type(candidates) is not int or not 6 <= candidates <= 30:
            raise ValueError("rerank candidates must be 6..30")
        self.base = retriever
        self.reranker = reranker if reranker is not None else LocalReranker()
        self.candidates, self.strict = candidates, strict
        self.last_status = 'not_run'
        self.disabled = False

    @property
    def manifest(self): return self.base.manifest
    @property
    def chunks(self): return self.base.chunks
    @property
    def vectors(self): return getattr(self.base, 'vectors', None)

    def search(self, question, top_k=6, alternate_query=None):
        baseline = self.base.search(question, top_k, alternate_query)
        if not baseline:
            self.last_status = 'no_hits'
            return baseline
        if self.disabled:
            self.last_status = 'fallback'
            return baseline
        try:
            pool = self.base.search(question, max(top_k, self.candidates), alternate_query)
            scores = np.asarray(self.reranker.score(question, pool, alternate_query), dtype=float)
            if scores.shape != (len(pool),) or not np.isfinite(scores).all():
                raise ValueError("Invalid reranker scores")
            order = sorted(range(len(pool)), key=lambda i: -scores[i])
            scope = source_scope(question) if getattr(self.base, 'use_scope', True) else set()
            # Preserve explicitly requested report coverage; score is not a probability.
            first = []
            if len(scope) > 1:
                for source in sorted(scope):
                    best = next((i for i in order if pool[i]['source_id'] == source), None)
                    if best is not None: first.append(best)
            order = first + [i for i in order if i not in first]
            pages, sources, selected = Counter(), Counter(), []
            limit = top_k if len(scope) == 1 else max(2, (top_k + 1) // 2)
            for i in order:
                hit = pool[i]; page = (hit['source_id'], hit['page'])
                if pages[page] >= 2 or sources[hit['source_id']] >= limit: continue
                selected.append(dict(hit, rerank_score=float(scores[i])))
                pages[page] += 1; sources[hit['source_id']] += 1
                if len(selected) >= top_k: break
            self.last_status = 'ok'
            return selected
        except Exception:
            if self.strict:
                raise RuntimeError("Reranker unavailable or invalid; evaluation stopped") from None
            self.disabled, self.last_status = True, 'fallback'
            return baseline
