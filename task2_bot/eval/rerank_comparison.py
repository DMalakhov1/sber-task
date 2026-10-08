"""Compare candidate coverage and final ranking; never call a paid API."""
import argparse
import json
import time
from pathlib import Path
from task2_bot.config import ROOT, INDEX
from task2_bot.rag.search import Retriever
from task2_bot.rag.rerank import LocalReranker, RerankingRetriever, DEFAULT_MODEL
from task2_bot.eval.retrieval import evaluate, load_questions


def compare(base, questions, reranker=None, candidates=24, translated=False):
    reports = {}
    for name, retriever, k in [('baseline', base, 6), ('candidates', base, candidates)]:
        start = time.perf_counter()
        reports[name] = evaluate(retriever, questions, k=k, translated=translated)
        reports[name]['elapsed_seconds'] = time.perf_counter() - start
    if reranker is not None:
        start = time.perf_counter()
        ranked = RerankingRetriever(base, reranker, candidates=candidates, strict=True)
        reports['reranked'] = evaluate(ranked, questions, translated=translated)
        reports['reranked']['elapsed_seconds'] = time.perf_counter() - start
    reports['note'] = ('Candidate coverage is not answer accuracy. Timings include cold model loading. '
                       'Manual translations do not evaluate the live translator. '
                       'Use dev for tuning; previously inspected test is not a fresh holdout.')
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['bm25','semantic','hybrid'], default='hybrid')
    p.add_argument('--split', choices=['dev','test'], default='dev')
    p.add_argument('--translation', choices=['off','manual'], default='off')
    p.add_argument('--candidates', type=int, default=24)
    p.add_argument('--candidate-only', action='store_true', help='No reranker model download')
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--device', default='cpu')
    p.add_argument('--index', type=Path, default=INDEX)
    p.add_argument('--questions', type=Path, default=ROOT/'eval'/'questions.json')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 6 <= a.candidates <= 30: p.error('candidates must be 6..30')
    if a.output.exists(): p.error('Output already exists; choose another filename')
    base = Retriever(a.index, mode=a.mode)
    questions = load_questions(a.questions, a.split)
    result = compare(base, questions,
                     None if a.candidate_only else LocalReranker(a.model, device=a.device),
                     a.candidates, a.translation == 'manual')
    result['configuration'] = dict(mode=a.mode, split=a.split, translation=a.translation,
        candidates=a.candidates, reranker=None if a.candidate_only else a.model,
        device=a.device, chunks_sha256=base.manifest['chunks_sha256'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({key: value['summary'] for key, value in result.items()
                      if isinstance(value, dict) and 'summary' in value}, ensure_ascii=False))


if __name__ == '__main__': main()
