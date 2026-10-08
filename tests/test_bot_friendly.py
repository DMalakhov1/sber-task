import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from task2_bot.bot.service import RAGService, NO_ANSWER
from task2_bot.bot.openai_client import APIError

HIT=dict(id='one',source_id='iea_electricity_2025',title='Electricity 2025',page=13,
         text='Electricity demand grew by 4.3% in 2024.',url='https://example.org/report.pdf')
class Search:
    vectors=None
    def search(self,*args):return [HIT]

@pytest.mark.parametrize('question,claim',[
    ('Как изменился спрос в 2024 году?','В 2024 году спрос вырос на 4,3%.'),
    ('А на сколько?','Рост составил 4,3%.'),
    ('Почему спрос упал в 2024?','В отчёте указан рост спроса на 4,3%, а не падение.')])
def test_friendly_style_keeps_evidence_and_single_generation(question,claim):
    answer={'abstain':False,'claims':[{'text':claim,'evidence':[{'id':'one','quote':HIT['text']}]}]}
    client=SimpleNamespace(complete=AsyncMock(return_value=answer))
    result=asyncio.run(RAGService(Search(),client,translation='off').answer_detailed(question,'SECRET'))
    assert result.reason=='ok' and '#page=13' in result.text and '[1]' in result.text
    assert client.complete.await_count==1
    prompt=client.complete.call_args.args[1][0]['content']
    assert 'доброжелательно' in prompt and 'предпосылку' in prompt


def test_missing_evidence_and_api_outage_are_different_and_secret_safe():
    client=SimpleNamespace(complete=AsyncMock(return_value={'abstain':True}))
    service=RAGService(Search(),client,translation='off')
    no=asyncio.run(service.answer_detailed('Вопрос','SECRET'))
    assert no.text==NO_ANSWER and no.reason=='abstain'
    client.complete.side_effect=APIError('SECRET and raw provider payload',reason='timeout')
    error=asyncio.run(service.answer_detailed('Вопрос','SECRET'))
    assert error.reason=='timeout' and 'не успел' in error.text and error.text!=no.text
    assert 'SECRET' not in error.text
