import hashlib
import json
from copy import deepcopy
from datetime import date
from pathlib import Path
from urllib.error import URLError
import pytest
from task3_excel.source_http import normalize, cached_request, request_json, TERRITORIES

FIXTURE = Path(__file__).parent/'fixtures'/'br_2026-09-01_excerpt.json'

def test_real_excerpt_territories_units_and_missing_ibr():
    data = json.loads(FIXTURE.read_text())
    assert TERRITORIES['Центр'] == 530000 and TERRITORIES['Восток'] == 540000
    rows = normalize(data['genconsum'], {'2026-09-01':data['hourly']}, date(2026,9,1), date(2026,9,1))
    by_system = {r['system']:r for r in rows}
    assert len(rows) == 2
    assert by_system['Центр']['generation_mwh'] == 22221
    assert by_system['Центр']['consumption_mwh'] == 23640
    assert by_system['Центр']['ibr_rub_mwh'] == 929.5288
    assert by_system['Восток']['ibr_rub_mwh'] is None
    assert by_system['Восток']['ibr_missing_reason'] == 'Источник не опубликовал ИБР'

def test_cache_detects_corruption_without_network(tmp_path):
    params = [('priceZone',''),('oesTerritory[]','530000')]
    raw = b'[]'
    (tmp_path/'x.json').write_bytes(raw)
    (tmp_path/'x.meta.json').write_text(json.dumps(dict(path='CommonInfo/GenConsum',params=[list(x) for x in params],sha256=hashlib.sha256(raw).hexdigest())))
    assert cached_request(tmp_path,'x','CommonInfo/GenConsum',params)[0] == []
    (tmp_path/'x.json').write_bytes(b'broken')
    with pytest.raises(ValueError, match='Повреждён кэш'):
        cached_request(tmp_path,'x','CommonInfo/GenConsum',params)

def test_real_excerpt_rejects_duplicate_hour_and_unknown_territory():
    data = json.loads(FIXTURE.read_text())
    data['genconsum'][0]['m_Item2'].append(dict(data['genconsum'][0]['m_Item2'][0]))
    with pytest.raises(ValueError, match='Дубли часов'):
        normalize(data['genconsum'], {'2026-09-01':data['hourly']}, date(2026,9,1), date(2026,9,1))
    data['genconsum'][0]['m_Item1'] = 999999
    with pytest.raises(ValueError, match='неизвестную территорию'):
        normalize(data['genconsum'], {'2026-09-01':data['hourly']}, date(2026,9,1), date(2026,9,1))

@pytest.mark.parametrize('change,reason', [
    (lambda data: data['genconsum'][0]['m_Item2'][0].update(INTERVAL=24), 'Недопустимый час'),
    (lambda data: data['hourly'][0]['m_Item2'][0].update(INTERVAL=True), 'Недопустимый час'),
    (lambda data: data['genconsum'][0]['m_Item2'][0].update(M_DATE='2026-08-31T00:00:00+03:00'), 'Неожиданная дата'),
    (lambda data: data['genconsum'][0]['m_Item2'][0].update(GEN_FACT=True), 'Некорректное число'),
    (lambda data: data['hourly'][0]['m_Item2'][0].update(AVERAGE_PRICE=float('nan')), 'Некорректное число'),
])
def test_real_excerpt_rejects_distorted_values(change, reason):
    data = json.loads(FIXTURE.read_text())
    change(data)
    with pytest.raises(ValueError, match=reason):
        normalize(data['genconsum'], {'2026-09-01':data['hourly']}, date(2026,9,1), date(2026,9,1))

def test_cache_parameter_mismatch_rejected(tmp_path):
    params = [('startDate','2026.09.01')]
    raw = b'[]'
    (tmp_path/'x.json').write_bytes(raw)
    (tmp_path/'x.meta.json').write_text(json.dumps(dict(path='CommonInfo/GetHourlyData',params=[list(x) for x in params],sha256=hashlib.sha256(raw).hexdigest())))
    with pytest.raises(ValueError, match='Повреждён кэш'):
        cached_request(tmp_path,'x','CommonInfo/GetHourlyData',[('startDate','2026.09.02')])

def test_timeout_retries_are_bounded(monkeypatch):
    calls = []
    class Opener:
        def open(self, request, timeout):
            calls.append(timeout)
            raise URLError(TimeoutError('test timeout'))
    monkeypatch.setattr('task3_excel.source_http.build_opener', lambda *args: Opener())
    monkeypatch.setattr('task3_excel.source_http.time.sleep', lambda delay: None)
    with pytest.raises(ValueError, match='TimeoutError'):
        request_json('CommonInfo/GetHourlyData', [], attempts=2)
    assert calls == [20, 20]
