"""Строгая проверка нормализованного почасового снимка и агрегация."""
import math
from datetime import datetime, timedelta

SYSTEMS = ('С-Запад', 'Центр', 'Юг', 'Ср. Волга', 'Урал', 'Сибирь', 'Восток')


def weighted_mean(pairs):
    pairs = list(pairs)
    if any(p is None or w is None for p, w in pairs):
        return None  # Неполный итог не выдаётся за полный.
    if any(type(p) not in (int, float) or type(w) not in (int, float)
           or not math.isfinite(p) or not math.isfinite(w) or w < 0 for p, w in pairs):
        raise ValueError('Некорректная цена или вес')
    denominator = math.fsum(w for _, w in pairs)
    return math.fsum(p * w for p, w in pairs) / denominator if denominator else None


def validate(rows, start, end, weight, allow_partial=False):
    if end < start:
        raise ValueError("Конец периода раньше начала")
    expected = set()
    day = start
    while day <= end:
        expected.update((day.isoformat(), hour, system) for hour in range(24) for system in SYSTEMS)
        day += timedelta(days=1)
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('date'), str) or not isinstance(row.get('system'), str):
            raise ValueError('Строка должна содержать строковые date и system')
        key = row['date'], row.get('hour'), row['system']
        datetime.strptime(row['date'], '%Y-%m-%d')
        if type(row['hour']) is not int or key not in expected:
            raise ValueError(f'Неожиданный интервал: {key}')
        if key in result:
            raise ValueError(f'Повтор интервала: {key}')
        if row.get('status') != 'actual':
            raise ValueError(f'Требуются фактические данные: {key}')
        for field in {'generation_mwh', 'consumption_mwh', 'ibr_rub_mwh', weight}:
            value = row.get(field)
            if value is None:
                reason = row.get(field.removesuffix('_mwh') + '_missing_reason') if field != 'ibr_rub_mwh' else row.get('ibr_missing_reason')
                if isinstance(reason, str) and reason.strip():
                    continue
                raise ValueError(f'Нет {field}: {key}')
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f'Не число в {field}: {key}')
            if field != 'ibr_rub_mwh' and value < 0:
                raise ValueError(f'Отрицательный объём: {key}')
        result[key] = row
    missing = expected - result.keys()
    if missing and not allow_partial:
        raise ValueError(f'Не хватает {len(missing)} интервалов; пример: {sorted(missing)[0]}')
    missing_prices = sum(r.get('ibr_rub_mwh') is None for r in result.values())
    missing_volumes = sum(r.get(field) is None for r in result.values() for field in ('generation_mwh', 'consumption_mwh'))
    if (missing_prices or missing_volumes) and not allow_partial:
        raise ValueError(f'Нет {missing_prices} значений ИБР и {missing_volumes} объёмов; для помеченного неполного отчёта укажите --allow-partial')
    for day, hour, system in sorted(missing):
        result[day, hour, system] = dict(date=day, hour=hour, system=system,
            generation_mwh=None, consumption_mwh=None, ibr_rub_mwh=None,
            **({weight: None} if weight not in ('generation_mwh', 'consumption_mwh') else {}),
            status='missing', ibr_missing_reason='Интервал не загружен',
            generation_missing_reason='Интервал не загружен', consumption_missing_reason='Интервал не загружен')
    return result


def complete_sum(values):
    values = list(values)
    return None if any(v is None for v in values) else math.fsum(values)


def quality_report(grid):
    missing = [dict(date=k[0], hour=k[1], system=k[2])
               for k, r in sorted(grid.items()) if r.get('status') == 'missing']
    prices = [dict(date=k[0], hour=k[1], system=k[2], reason=r['ibr_missing_reason'])
              for k, r in sorted(grid.items()) if r['ibr_rub_mwh'] is None]
    volumes = [dict(date=k[0], hour=k[1], system=k[2], field=field, reason=r.get(field.removesuffix('_mwh') + '_missing_reason'))
               for k, r in sorted(grid.items()) for field in ('generation_mwh', 'consumption_mwh') if r[field] is None]
    unpublished = [p for p in prices if grid[p['date'], p['hour'], p['system']].get('status') == 'actual']
    return dict(intervals_complete=not missing, volumes_complete=not volumes,
                prices_complete=not prices, unpublished_prices=unpublished,
                expected_records=len(grid), observed_records=len(grid)-len(missing),
                complete=not missing and not prices and not volumes, missing_intervals=missing,
                missing_prices=prices, missing_volumes=volumes)


def published_price_summary(records, weight):
    """Explicit partial-scope analytics; never substitute for the seven-system total."""
    records = list(records)
    available = [(r['ibr_rub_mwh'], r.get(weight)) for r in records
                 if r['ibr_rub_mwh'] is not None and r.get(weight) is not None]
    prices = [r['ibr_rub_mwh'] for r in records if r['ibr_rub_mwh'] is not None]
    denominator = complete_sum(r.get(weight) for r in records)
    covered_weight = math.fsum(w for _, w in available)
    return dict(weighted=weighted_mean(available),
                arithmetic=math.fsum(prices)/len(prices) if prices else None,
                published_count=len(prices),
                weight_coverage=covered_weight/denominator if denominator else None)
