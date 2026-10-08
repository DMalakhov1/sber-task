import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import numpy as np
from task2_bot.config import INDEX
from task2_bot.rag.download import sha256


def tokens(text):
    return re.findall(r"[\w]+", text.casefold(), flags=re.UNICODE)

def source_scope(question):
    text = question.casefold()
    patterns = {
        "iea_electricity_2025": ("electricity 2025", "electricity2025", "iea_electricity_2025"),
        "iea_global_energy_review_2025": ("global energy review", "globalenergyreview", "iea_global_energy_review_2025"),
        "sipr_2025_2030": ("сипр", "си пр", "sipr_2025_2030"),
    }
    return {source for source, aliases in patterns.items() if any(alias in text for alias in aliases)}


class Retriever:
    def __init__(self, directory=INDEX, embedder=None, mode="hybrid", use_scope=True, lexical_weight=1.0):
        if mode not in {'bm25','semantic','hybrid'}: raise ValueError('Unknown search mode')
        if not 0 <= lexical_weight <= 1: raise ValueError('lexical_weight must be 0..1')
        self.mode, self.use_scope, self.lexical_weight = mode, use_scope, lexical_weight
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        m = self.manifest
        if m.get("version") != 1: raise ValueError("Неизвестная версия индекса")
        def checked_file(name, digest):
            path = (directory / name).resolve()
            if path.parent != directory.resolve() or sha256(path) != digest:
                raise ValueError("Повреждён индекс или небезопасный путь")
            return path
        self.chunks = json.loads(checked_file(m['chunks_file'], m['chunks_sha256']).read_text())
        if not isinstance(self.chunks, list) or not self.chunks or len(self.chunks) != m['count']:
            raise ValueError('Пустой или неполный индекс')
        required = ('id','source_id','text','title','url')
        if any(not isinstance(c, dict) or any(not isinstance(c.get(k), str) or not c[k].strip() for k in required)
               or type(c.get('page')) is not int or c['page'] < 1 for c in self.chunks):
            raise ValueError('Некорректный фрагмент индекса')
        if len({c['id'] for c in self.chunks}) != len(self.chunks):
            raise ValueError('Повтор chunk_id в индексе')
        self.vectors = None
        self.embedder = embedder
        if m.get('vectors_file') and mode != 'bm25':
            self.vectors = np.load(checked_file(m['vectors_file'], m['vectors_sha256']), allow_pickle=False)
            if (self.vectors.ndim != 2 or len(self.vectors) != len(self.chunks)
                    or self.vectors.shape[1] == 0 or not np.isfinite(self.vectors).all()
                    or np.any(np.linalg.norm(self.vectors, axis=1) < 1e-12)):
                raise ValueError("Размерность индекса не совпадает")
            if embedder is None:
                from task2_bot.rag.embeddings import Embedder
                self.embedder = Embedder(model=m['model'])
        if mode == "semantic" and self.vectors is None: raise ValueError("Нужен семантический индекс")
        self.counts = [Counter(tokens(c['text'])) for c in self.chunks]
        self.lengths = np.array([sum(c.values()) for c in self.counts])
        self.avg_length = max(float(self.lengths.mean()), 1)
        self.df = Counter(t for c in self.counts for t in c)

    def _query_vector(self, text):
        vector = np.asarray(self.embedder.query(text), dtype=float)
        if (vector.ndim != 1 or vector.shape[0] != self.vectors.shape[1]
                or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-12):
            raise ValueError('Эмбеддинг запроса не соответствует индексу')
        return vector / np.linalg.norm(vector)

    def search(self, question, top_k=6, alternate_query=None):
        if not question.strip() or top_k < 1: return []
        scope = source_scope(question) if self.use_scope else set()
        candidates = [i for i,c in enumerate(self.chunks) if not scope or c['source_id'] in scope]
        if not candidates: return []
        scores = np.zeros(len(self.chunks))
        for term in set(tokens(question + " " + (alternate_query or ""))):
            idf = math.log(1 + (len(scores) - self.df[term] + .5) / (self.df[term] + .5))
            tf = np.array([c[term] for c in self.counts])
            scores += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * self.lengths / self.avg_length))
        ranks = np.zeros(len(scores))
        for rank, idx in enumerate(sorted(candidates, key=lambda i: -scores[i])[:30]):
            if scores[idx] > 0 and self.mode != 'semantic':
                ranks[idx] += (self.lexical_weight if self.vectors is not None else 1) / (60 + rank + 1)
        semantic = None
        if self.vectors is not None:
            q = self._query_vector(question)
            semantic = self.vectors @ q
            if alternate_query:
                alt = self._query_vector(alternate_query)
                semantic = np.maximum(semantic, self.vectors @ alt)
            for rank, idx in enumerate(sorted(candidates, key=lambda i: -semantic[i])[:30]):
                ranks[idx] += 1 / (60 + rank + 1)
        results, per_page, per_source = [], Counter(), Counter()
        ordered = sorted(candidates, key=lambda i: -ranks[i])
        if len(scope) > 1:
            # At least one best candidate per explicitly named report before the rest.
            first = []
            for source in sorted(scope):
                best = next((i for i in ordered if self.chunks[i]['source_id'] == source and ranks[i] > 0), None)
                if best is not None: first.append(best)
            ordered = first + [i for i in ordered if i not in first]
        source_limit = top_k if len(scope) == 1 else max(2, (top_k + 1) // 2)
        for idx in ordered:
            if ranks[idx] <= 0: continue
            chunk = self.chunks[idx]
            key = (chunk['source_id'], chunk['page'])
            if per_page[key] >= 2 or per_source[chunk['source_id']] >= source_limit: continue
            per_page[key] += 1
            per_source[chunk['source_id']] += 1
            results.append(dict(chunk, score=float(ranks[idx]),
                                semantic_score=float(semantic[idx]) if semantic is not None else None))
            if len(results) == top_k: break
        return results


def main():
    p = argparse.ArgumentParser(description="Локальный поиск; API не расходуется")
    p.add_argument("question")
    p.add_argument("--index", type=Path, default=INDEX)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--query-en", help="Английская формулировка для межъязыкового поиска; без API")
    a = p.parse_args()
    for hit in Retriever(a.index).search(a.question, a.top_k, a.query_en):
        print(f"{hit['id']} | {hit['title']} | PDF стр. {hit['page']}\n{hit['text']}\n{hit['url']}\n")

if __name__ == "__main__": main()
