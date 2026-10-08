import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from task2_bot.config import ROOT, INDEX
from task2_bot.rag.search import Retriever


def load_questions(path, split):
    questions = json.loads(Path(path).read_text())
    # Split membership is controlled by checked-in manifest, not the evaluation invocation.
    manifest = ROOT / 'eval' / 'split_manifest.json'
    if Path(path).resolve() == (ROOT / 'eval' / 'questions.json').resolve():
        m = json.loads(manifest.read_text())
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != m['questions_sha256']:
            raise ValueError('Набор изменён после фиксации split. Создайте новую версию набора и manifest.')
        for q in questions:
            if m['groups'][q['group_id']] != q['split']: raise ValueError('Нарушено разделение групп')
    return [q for q in questions if q['split'] == split]


def counts(rows):
    n = len(rows)
    return {'n': n, 'page_hits': sum(x['hit'] for x in rows),
            'all_sources_hit': sum(x['all_sources'] for x in rows),
            'all_evidence_pages_hit': sum(x['all_pages'] for x in rows),
            'mrr': sum(1/x['rank'] if x['rank'] else 0 for x in rows)/n if n else None}


def evaluate(retriever, questions, k=6, translated=False):
    rows, skipped = [], []
    for q in questions:
        gold = q.get('retrieval_gold', [])
        if q.get('status') != 'gold_verified' or not gold or q['kind'] == 'injection':
            skipped.append({'id':q['id'],'reason':'not_retrieval_gold'}); continue
        for source_id, digest in q.get('corpus_sha256', {}).items():
            actual = retriever.manifest.get('metadata', {}).get(source_id, {}).get('sha256')
            if actual != digest: raise ValueError('Gold and index corpus hashes differ')
        hits = retriever.search(q['question'], k, q.get('query_en') if translated else None)
        correct = lambda h,e: h['source_id']==e['source_id'] and h['page'] in e['pages']
        rank = next((i for i,h in enumerate(hits,1) if any(correct(h,e) for e in gold)), None)
        group = 'ru_to_ru' if all(e['source_id']=='sipr_2025_2030' for e in gold) else (
            'ru_to_en' if all(e['source_id']!='sipr_2025_2030' for e in gold) else 'cross_language')
        rows.append({'id':q['id'],'group':group,'translation_used':bool(translated and q.get('query_en')),'hit':rank is not None,'rank':rank,
            'all_sources':all(any(h['source_id']==e['source_id'] for h in hits) for e in gold),
            'all_pages':all(any(correct(h,e) for h in hits) for e in gold),
            'retrieved':[{'id':h['id'],'source':h['source_id'],'page':h['page']} for h in hits]})
    groups = sorted({x['group'] for x in rows})
    return dict(k=k, summary=counts(rows), groups={g:counts([x for x in rows if x['group']==g]) for g in groups},
                rows=rows,skipped=skipped,translation='manual' if translated else 'off',
                note='Counts are numerator/denominator. Manual translations are not measurements of the live translator.')


def main():
    p=argparse.ArgumentParser(description='Оценка retrieval; платных вызовов нет')
    p.add_argument('--index',type=Path,default=INDEX)
    p.add_argument('--questions',type=Path,default=ROOT/'eval'/'questions.json')
    p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--mode',choices=['bm25','semantic','hybrid'],default='hybrid')
    p.add_argument('--translation',choices=['off','manual'],default='off')
    p.add_argument('--scope',choices=['on','off'],default='on')
    p.add_argument('--lexical-weight',type=float,default=1.0)
    p.add_argument('--output',type=Path,default=ROOT/'outputs'/'retrieval_review.json')
    a=p.parse_args()
    retriever=Retriever(a.index,mode=a.mode,use_scope=a.scope=='on',lexical_weight=a.lexical_weight)
    result=evaluate(retriever,load_questions(a.questions,a.split),translated=a.translation=='manual')
    result['configuration']={'split':a.split,'mode':a.mode,'scope':a.scope,'lexical_weight':a.lexical_weight,
                              'model':retriever.manifest.get('model'),'chunks_sha256':retriever.manifest['chunks_sha256']}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'configuration':result['configuration'],'summary':result['summary'],'groups':result['groups'],
                      'skipped':len(result['skipped'])},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
