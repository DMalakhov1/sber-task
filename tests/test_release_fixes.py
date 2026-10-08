import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import httpx
import pytest
from task2_bot.bot.service import RAGService
from task2_bot.bot.openai_client import OpenAIClient
from task2_bot.bot.deepseek_client import DeepSeekClient
from task3_excel.source_http import request_json

HIT = dict(id='c1', source_id='iea', text='Global electricity demand grew by 4.3% in 2024.',
           title='IEA', page=13, url='https://example.org/report.pdf')
PAYLOAD = dict(abstain=False, claims=[dict(text='Спрос упал на 99%.',
    evidence=[dict(id='c1', quote=HIT['text'])])])

@pytest.mark.parametrize('review,expected', [({'supported':[False]},'unsupported_claim'),
    ({'supported':[]},'verification_invalid'), ({'supported':['true']},'verification_invalid'),
    ({'supported':[True]},'ok')])
def test_semantic_review_fail_closed_and_bounded(review, expected):
    client=SimpleNamespace(complete=AsyncMock(side_effect=[PAYLOAD, review]))
    service=RAGService(SimpleNamespace(search=lambda q,k:[HIT]),client,translation='off',verify_claims=True)
    result=asyncio.run(service.answer_detailed('Что со спросом?','SECRET'))
    assert result.reason == expected
    assert client.complete.await_count == 2
    if expected != 'ok': assert '99%' not in result.text
    messages=client.complete.call_args.args[1]
    assert 'SECRET' not in json.dumps(messages)
    assert json.loads(messages[1]['content'])['evidence'][0]['text'] == HIT['text']

def test_no_semantic_review_for_abstention():
    client=SimpleNamespace(complete=AsyncMock(return_value={'abstain':True}))
    result=asyncio.run(RAGService(SimpleNamespace(search=lambda q,k:[HIT]),client,translation='off',verify_claims=True).answer_detailed('Вопрос','SECRET'))
    assert result.reason == 'abstain' and client.complete.await_count == 1

def test_default_official_provider_and_explicit_alternative():
    assert OpenAIClient('deepseek-flash').base_url == 'https://api.deepseek.com'
    assert DeepSeekClient().base_url == 'https://api.deepseek.com'
    assert OpenAIClient('test',base_url='https://example.org/v1').base_url == 'https://example.org/v1'

def test_read_timeout_is_retried(monkeypatch):
    calls=[]
    class Opener:
        def open(self, request, timeout):
            calls.append(timeout)
            raise TimeoutError('read timed out')
    monkeypatch.setattr('task3_excel.source_http.build_opener',lambda *a:Opener())
    monkeypatch.setattr('task3_excel.source_http.time.sleep',lambda delay:None)
    with pytest.raises(ValueError,match='Таймаут'):
        request_json('CommonInfo/GetHourlyData',[],attempts=3)
    assert calls == [20,20,20]

def test_partial_and_error_exit_codes_differ(tmp_path,monkeypatch):
    from task3_excel import export_excel
    path=Path(__file__).resolve().parents[1]/'task3_excel/data/processed/2026-09.json'
    import sys
    args=['export','--month','2026-09','--input',str(path),'--weight','consumption_mwh','--allow-partial','--output',str(tmp_path/'result.xlsx')]
    monkeypatch.setattr(sys,'argv',args)
    assert export_excel.main() == 3
    assert (tmp_path/'result.xlsx').exists()
    with pytest.raises(SystemExit) as exc:export_excel.main()
    assert exc.value.code == 2
