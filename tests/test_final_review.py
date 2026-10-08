import json
from datetime import date
import pytest
from openpyxl import load_workbook
from task3_excel.aggregation import validate, quality_report, SYSTEMS
from task3_excel.export_excel import export
from tools.probe_br_site import wsdl_summary


def day_snapshot():
    return {'metadata': dict(source_url='https://br.so-ups.ru/', downloaded_at='2026-10-06',
        timezone='Europe/Moscow', hour_convention='interval_start_0_23', energy_unit='MWh',
        price_unit='RUB/MWh', weight_basis='consumption_mwh', weight_explanation='synthetic test'),
        'rows': [dict(date='2026-09-01', hour=h, system=s, status='actual',
            generation_mwh=10, consumption_mwh=5, ibr_rub_mwh=100) for h in range(24) for s in SYSTEMS]}


def test_partial_workbook_does_not_publish_incomplete_totals(tmp_path):
    data=day_snapshot(); data['rows'].pop(0)
    source=tmp_path/'day.json'; source.write_text(json.dumps(data))
    output=tmp_path/'partial.xlsx'; day=date(2026,9,1)
    with pytest.raises(ValueError): export(source, output, day, day, 'consumption_mwh')
    export(source, output, day, day, 'consumption_mwh', allow_partial=True)
    book=load_workbook(output,data_only=True)
    assert len(book.worksheets)==3
    for sheet in book:
        assert 'НЕПОЛНЫЕ' in sheet['A2'].value
        assert sheet['C5'].value is None and sheet['C5'].comment
        assert sheet['J5'].value is None
        assert sheet['C29'].value is None and sheet['J29'].value is None
        assert sheet['D29'].value is not None
    quality=json.loads(output.with_suffix('.quality.json').read_text())
    assert quality['observed_records']==167 and quality['expected_records']==168
    assert len(quality['missing_intervals'])==1 and len(quality['xlsx_sha256'])==64


def test_partial_still_rejects_invalid_values():
    rows=day_snapshot()['rows']; rows[0]['generation_mwh']=-1
    with pytest.raises(ValueError): validate(rows,date(2026,9,1),date(2026,9,1),'consumption_mwh',True)


def test_empty_partial_is_not_zero():
    grid=validate([],date(2026,9,1),date(2026,9,1),'balance_volume_mwh',True)
    assert len(grid)==168 and not quality_report(grid)['complete']
    assert all(r['balance_volume_mwh'] is None for r in grid.values())


def test_wsdl_operations_and_reject_html_entities():
    body=b'<definitions xmlns="http://schemas.xmlsoap.org/wsdl/"><portType><operation name="Example"/></portType></definitions>'
    assert wsdl_summary(body)['operations']==['Example']
    for body in [b'<html/>',b'<!DOCTYPE x><x/>']:
        with pytest.raises(ValueError): wsdl_summary(body)


def test_block_interval_respects_effective_sample_and_pairing():
    from task1_brent.review_experiments import block_interval
    assert block_interval([-1]*30,21)['status']=='insufficient_blocks'
    result=block_interval([-2]*60,12)
    assert result['ci95']==[-2,-2]
    assert result==block_interval([-2]*60,12)


@pytest.mark.parametrize('enabled,valid_second,expected_calls,reason',[(False,True,1,'quote_mismatch'),(True,True,2,'ok'),(True,False,2,'quote_mismatch')])
def test_quote_repair_is_optional_bounded_and_still_validated(enabled,valid_second,expected_calls,reason):
    import asyncio
    from task2_bot.bot.service import RAGService
    text='The report states that total electricity demand increased in 2024.'
    hit=dict(id='one',text=text,title='Report',page=1,url='https://example.org/report.pdf')
    class Client:
        calls=0
        async def complete(self,key,messages):
            self.calls+=1
            quote=text if self.calls==2 and valid_second else 'This quotation is invented and absent from evidence.'
            return {'abstain':False,'claims':[{'text':'Спрос вырос.','evidence':[{'id':'one','quote':quote}]}]}
    client=Client()
    result=asyncio.run(RAGService(None,client,repair_quotes=enabled).answer_detailed('Вопрос','fake',context_override=[hit]))
    assert result.reason==reason and client.calls==expected_calls
    assert result.repair_attempted==enabled
