"""Проверка нормализованного снимка; контракт API описан в docs/SITE_RECON.md."""
import hashlib
import json
from pathlib import Path

WEIGHTS = {'consumption_mwh', 'generation_mwh', 'balance_volume_mwh'}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Повтор поля JSON: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Недопустимое значение JSON: {value}')


def load_snapshot(path):
    path = Path(path)
    raw = path.read_bytes()
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f'Повреждён JSON-снимок {path.name}: {exc}') from None
    if not isinstance(data, dict) or not isinstance(data.get('metadata'), dict):
        raise ValueError('Снимок должен содержать объект metadata и список rows')
    meta = data['metadata']
    required = ('source_url', 'downloaded_at', 'timezone', 'hour_convention', 'energy_unit',
                'price_unit', 'weight_basis', 'weight_explanation')
    if any(not isinstance(meta.get(key), str) or not meta[key].strip() for key in required):
        raise ValueError('Метаданные источника и методики должны быть непустыми строками')
    if meta['weight_basis'] not in WEIGHTS:
        raise ValueError('Неизвестная основа взвешивания ИБР')
    if 'synthetic' in meta and type(meta['synthetic']) is not bool:
        raise ValueError('synthetic должен быть логическим значением')
    if meta['hour_convention'] != 'interval_start_0_23':
        raise ValueError('Нужны часы начала интервалов 0–23')
    if meta['energy_unit'] != 'MWh' or meta['price_unit'] != 'RUB/MWh':
        raise ValueError('Требуется явная нормализация единиц в MWh и RUB/MWh')
    if meta['timezone'] != 'Europe/Moscow':
        raise ValueError('Требуется предварительное приведение к московскому времени')
    rows = data.get('rows')
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError('rows должен быть списком объектов')
    return rows, meta, hashlib.sha256(raw).hexdigest()
