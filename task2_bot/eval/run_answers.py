"""Opt-in, capped paid experiments. Reads key with getpass; never accepts it as CLI argument."""
import argparse
import asyncio
from dataclasses import asdict
import getpass
import json
import time
from pathlib import Path
from task2_bot.config import ROOT, INDEX
from task2_bot.bot.openai_client import APIError, OpenAIClient
from task2_bot.bot.service import RAGService
from task2_bot.bot.telemetry import collect_usage
from task2_bot.rag.search import Retriever
from task2_bot.eval.retrieval import load_questions

class BudgetClient(OpenAIClient):
    def __init__(self, max_calls, **kwargs):
        kwargs.setdefault("model", "test-model")
        super().__init__(**kwargs);self.remaining=max_calls
    async def complete(self, *args, **kwargs):
        if self.remaining<=0: raise APIError('Лимит вызовов опыта исчерпан.')
        self.remaining-=1
        return await super().complete(*args, **kwargs)


def altered_context(hits, variant, chunks, q):
    copied=[dict(h) for h in hits]
    if variant=='injection':
        injection=q.get('injection')
        if not injection or not copied: raise ValueError('Для этого вопроса нет injection fixture')
        copied[0]['text']=injection['instruction']+' '+injection['marker']+'\n'+copied[0]['text']
    elif variant=='counterfactual':
        # Explicit fixture, never edits the on-disk index or the PDF.
        spec=q.get('counterfactual')
        if not spec: raise ValueError('Нет явного counterfactual fixture')
        matches=[h for h in copied if spec['old'] in h['text']]
        if not matches: raise ValueError('Заменяемое число не найдено в выбранных фрагментах')
        for h in copied: h['text']=h['text'].replace(spec['old'],spec['new'])
    elif variant=='wrong_fragment':
        wanted={e['source_id'] for e in q.get('retrieval_gold',[])}
        unrelated=next((dict(h) for h in chunks if h['source_id'] not in wanted and len(h['text'])>100),None)
        if unrelated is None: raise ValueError('Не найден чужой фрагмент')
        copied=[unrelated]
    return copied


async def experiment(args):
    questions=load_questions(args.questions,args.split)
    if args.ids:
        requested=set(args.ids.split(','));questions=[q for q in questions if q['id'] in requested]
        if {q['id'] for q in questions}!=requested: raise ValueError('ID не входят в выбранный split')
    questions=questions[:args.max_questions]
    if not questions: raise ValueError('Нет вопросов')
    key=getpass.getpass('OpenAI-compatible API key (не сохраняется): ').strip()
    if not key: raise ValueError('Ключ не введён')
    client=BudgetClient(args.max_api_calls,model=args.model,base_url=args.base_url)
    retriever=Retriever(args.index)
    if args.rerank == 'on':
        from task2_bot.rag.rerank import RerankingRetriever
        retriever = RerankingRetriever(retriever, strict=True)
    service=RAGService(retriever,client,translation=args.translation,
        verify_claims=args.verify_claims == "on", verify_relevance=args.verify_relevance == "on")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Refuse to overwrite an earlier experiment.
    try:
        with args.output.open('x',encoding='utf-8') as out:
            for q in questions:
                if client.remaining<=0: break
                started = time.perf_counter()
                if args.variant=='no_rag':
                    with collect_usage() as usage:
                        try:
                            payload=await client.complete(key,[{'role':'system','content':'Ты эксперт по электроэнергетике. '
                                'Ответь кратко по-русски. Документы не предоставлены. Если не знаешь, признай это. '
                                'Верни JSON {"answer":"текст"}.'},{'role':'user','content':q['question']}])
                            record={'text':payload.get('answer',''),'reason':'baseline_unverified','usage':asdict(usage)}
                        except APIError as exc: record={'text':'','reason':exc.reason,'usage':asdict(usage)}
                else:
                    override=None
                    if args.variant!='rag':
                        hits=await asyncio.to_thread(retriever.search,q['question'],6,q.get('query_en'))
                        try: override=altered_context(hits,args.variant,retriever.chunks,q)
                        except ValueError:
                            record={'id':q['id'],'variant':args.variant,'reason':'fixture_unavailable'}
                            out.write(json.dumps(record)+'\n');continue
                    result=await service.answer_detailed(q['question'],key,context_override=override)
                    record={'text':result.text,'claims':result.claims,'reason':result.reason,'trace':result.trace,
                            'usage':asdict(result.usage),'latency_ms':result.latency_ms,
                            'translation_fallback':result.translation_fallback,'rerank_status':result.rerank_status,
                            'canary_leaked':bool(q.get('injection',{}).get('marker') and q['injection']['marker'] in result.text)}
                    if override is not None: record['synthetic_context']=override
                record.setdefault('latency_ms', (time.perf_counter()-started)*1000)
                record.update(id=q['id'],split=q['split'],variant=args.variant, configuration={
                    'rerank':args.rerank,'verify_relevance':args.verify_relevance,'api_model':args.model,'api_base_url':args.base_url,'verify_claims':args.verify_claims,'embedding_model':retriever.manifest.get('model'),
                    'chunks_sha256':retriever.manifest['chunks_sha256'],'translation':args.translation,
                    'context_selection':'manual_translation' if args.variant in {'injection','counterfactual','wrong_fragment'} else 'runtime'})
                out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush()
    finally:
        key='';await client.close()
    print('Результаты записаны. Это данные для ручной оценки, не автоматическое подтверждение точности.')


def main():
    p=argparse.ArgumentParser(description='Платные контрольные опыты: только локально с вашим ключом')
    p.add_argument('--allow-paid',action='store_true')
    p.add_argument('--max-api-calls',type=int,default=4)
    p.add_argument('--max-questions',type=int,default=2)
    p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--ids')
    p.add_argument('--variant',choices=['rag','no_rag','counterfactual','wrong_fragment','injection'],default='rag')
    p.add_argument('--translation',choices=['on','off'],default='on')
    p.add_argument('--model',default='deepseek-flash',help='Модель API')
    p.add_argument('--base-url', default='https://api.deepseek.com')
    p.add_argument('--rerank', choices=['on','off'], default='off')
    p.add_argument('--verify-relevance', choices=['on','off'], default='off')
    p.add_argument('--verify-claims', choices=['on','off'], default='on')
    p.add_argument('--questions',type=Path,default=ROOT/'eval'/'questions.json')
    p.add_argument('--index',type=Path,default=INDEX)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.allow_paid: p.error('Для реальных вызовов явно укажите --allow-paid и лимит --max-api-calls')
    if not 1<=a.max_api_calls<=100 or not 1<=a.max_questions<=40: p.error('Некорректные лимиты')
    asyncio.run(experiment(a))

if __name__=='__main__': main()
