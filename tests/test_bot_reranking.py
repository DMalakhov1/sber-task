import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import numpy as np
import pytest
from task2_bot.rag.rerank import LocalReranker, RerankingRetriever
from task2_bot.bot.service import RAGService


def hit(i, source='iea_electricity_2025', page=None):
    return dict(id=str(i), source_id=source, page=page or i+1, title='Report',
                text='Electricity demand grew by 4.3% in 2024.',
                url='https://example.org/report.pdf', score=1/(i+1), semantic_score=None)


class Base:
    use_scope = True
    manifest = {}
    vectors = None
    def __init__(self, hits): self.chunks = hits
    def search(self, question, top_k=6, alternate_query=None): return self.chunks[:top_k]


def test_rerank_promotes_candidate_and_preserves_provenance():
    hits = [hit(i) for i in range(24)]
    original = copy.deepcopy(hits)
    scorer = SimpleNamespace(score=lambda *args: np.arange(24))
    ranked = RerankingRetriever(Base(hits), scorer)
    result = ranked.search('Electricity 2025 спрос')
    assert [h['id'] for h in result] == ['23','22','21','20','19','18']
    assert result[0]['rerank_score'] == 23
    assert {k:v for k,v in result[0].items() if k != 'rerank_score'} == hits[23]
    assert hits == original and ranked.last_status == 'ok'


@pytest.mark.parametrize('bad', [np.full(24, np.nan), np.full(24, np.inf), [1], [[1]]*24])
def test_invalid_scores_fall_back_without_changing_baseline(bad):
    base = Base([hit(i) for i in range(24)])
    scorer = SimpleNamespace(score=lambda *args: bad)
    ranked = RerankingRetriever(base, scorer)
    assert ranked.search('Electricity 2025') == base.search('Electricity 2025')
    assert ranked.last_status == 'fallback' and ranked.disabled


def test_failed_model_is_not_reloaded_for_every_user():
    calls = []
    def fail(*args):
        calls.append(1)
        raise RuntimeError('provider detail should not leak')
    ranked = RerankingRetriever(Base([hit(i) for i in range(8)]), SimpleNamespace(score=fail))
    for _ in range(3): assert ranked.search('question')
    assert len(calls) == 1
    strict = RerankingRetriever(ranked.base, SimpleNamespace(score=fail), strict=True)
    with pytest.raises(RuntimeError, match='evaluation stopped'): strict.search('question')


def test_explicit_reports_and_page_diversity_survive_reranking():
    hits = [hit(i, page=1 if i<5 else i) for i in range(12)]
    hits += [hit(20+i, 'sipr_2025_2030') for i in range(12)]
    ranked = RerankingRetriever(Base(hits), SimpleNamespace(score=lambda *a: np.arange(24)))
    result = ranked.search('Сравни СиПР и Electricity 2025')
    assert len(result) == 6
    assert sum(h['source_id']=='sipr_2025_2030' for h in result) == 3
    assert sum(h['source_id']=='iea_electricity_2025' for h in result) == 3


def test_local_model_receives_language_specific_pairs():
    seen = []
    class Model:
        def predict(self, pairs, **kwargs):
            seen.extend(pairs)
            return np.array([0., 1.])
    scores = LocalReranker(model=Model()).score('Русский вопрос',
        [hit(0), hit(1,'sipr_2025_2030')], 'English question')
    assert scores.tolist() == [0., 1.]
    assert seen[0][0] == 'English question' and seen[1][0] == 'Русский вопрос'


@pytest.mark.parametrize('relevant,reason', [(True,'ok'), (False,'off_topic_answer'),
                                           ('yes','verification_invalid'), (None,'verification_invalid')])
def test_relevance_shares_one_verifier_call_and_passes_question(relevant, reason):
    h = hit(0)
    answer = {'abstain':False, 'claims':[{'text':'Спрос вырос на 4,3%.',
                'evidence':[{'id':h['id'], 'quote':h['text']}]}]}
    review = {'supported':[True], 'answers_question':relevant}
    client = SimpleNamespace(complete=AsyncMock(side_effect=[answer, review]))
    service = RAGService(Base([h]), client, translation='off', verify_claims=True, verify_relevance=True)
    result = asyncio.run(service.answer_detailed('А в 2025 году?', 'SECRET',
               history=[{'question':'Как вырос спрос в 2024?','answer':'Старый ответ','claims':['Спрос вырос в 2024 году.']}]))
    assert result.reason == reason and client.complete.await_count == 2
    payload = json.loads(client.complete.call_args.args[1][1]['content'])
    assert payload['question'] == 'А в 2025 году?'
    ctx = payload['conversation_context'][0]
    # Теперь в контекст попадают только проверенные утверждения, а не сырой прошлый ответ.
    assert ctx['question'] == 'Как вырос спрос в 2024?' and ctx['claims'] == ['Спрос вырос в 2024 году.']
    assert 'answer' not in ctx and 'Старый ответ' not in json.dumps(payload)
    assert 'SECRET' not in json.dumps(payload)
    if reason != 'ok': assert '4,3%' not in result.text


def test_off_switch_does_not_call_judge():
    client = SimpleNamespace(complete=AsyncMock(return_value={'abstain':True}))
    result = asyncio.run(RAGService(Base([hit(0)]),client,translation='off').answer_detailed('Question','SECRET'))
    assert client.complete.await_count == 1 and result.rerank_status == 'off'


def test_empty_search_does_not_load_model():
    def fail(*a): raise AssertionError('must not load')
    ranked = RerankingRetriever(Base([]), SimpleNamespace(score=fail))
    assert ranked.search('q') == [] and ranked.last_status == 'no_hits'


def test_fallback_is_visible_in_service_metrics_without_exception_text(tmp_path):
    from task2_bot.bot.telemetry import MetricSink
    h = hit(0)
    def fail(*a): raise RuntimeError('SECRET-ERROR')
    ranked = RerankingRetriever(Base([h]), SimpleNamespace(score=fail))
    client = SimpleNamespace(complete=AsyncMock(return_value={'abstain':True}))
    path = tmp_path/'metrics.jsonl'
    service = RAGService(ranked, client, translation='off', metrics=MetricSink(path))
    result = asyncio.run(service.answer_detailed('question', 'SECRET-KEY'))
    assert result.rerank_status == 'fallback'
    row = json.loads(path.read_text())
    assert row['rerank_status'] == 'fallback'
    assert 'SECRET' not in path.read_text()


def test_page_cap_survives_reranking():
    hits = [hit(i, page=1 if i < 5 else i+1) for i in range(10)]
    ranked = RerankingRetriever(Base(hits), SimpleNamespace(score=lambda *a: np.arange(10)[::-1]))
    result = ranked.search('Electricity 2025')
    assert len(result) == 6
    assert sum(h['page'] == 1 for h in result) == 2
