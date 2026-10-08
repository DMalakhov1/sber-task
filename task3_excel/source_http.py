"""Публичные GET-данные СБР → проверяемый снимок; без cookies и авторизации."""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, build_opener, HTTPSHandler, HTTPRedirectHandler
from zoneinfo import ZoneInfo
import tempfile

from .aggregation import SYSTEMS, validate
from .export_excel import parse_month, previous_month, export

BASE = 'https://br.so-ups.ru/webapi/api/'
TERRITORIES = {'С-Запад': 840000, 'Центр': 530000, 'Юг': 550000,
               'Ср. Волга': 600000, 'Урал': 630000, 'Сибирь': 610000, 'Восток': 540000}
MAX_BODY = 2_000_000

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


def request_json(path, params, *, insecure=False, attempts=3):
    if not path.startswith(('dict/', 'CommonInfo/')):
        raise ValueError('Недопустимый API-маршрут')
    context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
    opener = build_opener(HTTPSHandler(context=context), NoRedirect())
    url = BASE + path + ('?' + urlencode(params, doseq=True) if params else '')
    for attempt in range(attempts):
        try:
            with opener.open(Request(url, headers={'User-Agent':'SberAnalystTask3/1.0'}), timeout=20) as response:
                raw = response.read(MAX_BODY + 1)
                if len(raw) > MAX_BODY: raise ValueError('Ответ источника превышает ограничение 2 МБ')
                try: data = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f'Источник вернул некорректный JSON для {path}') from exc
                return raw, data
        except HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise ValueError(f'HTTP {exc.code} для {path}; проверьте параметры или повторите позже') from None
            retry = exc.headers.get('Retry-After', '')
            delay = int(retry) if retry.isdigit() else 2 ** attempt
            time.sleep(min(delay, 30))
        except TimeoutError:
            if attempt == attempts - 1:
                raise ValueError(f'Таймаут br.so-ups.ru для {path}; повторите позже') from None
            time.sleep(2 ** attempt)
        except URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise ValueError('Сертификат br.so-ups.ru не прошёл проверку; только для этого хоста доступен --insecure-br') from None
            if attempt == attempts - 1:
                raise ValueError(f'Не удалось соединиться с br.so-ups.ru для {path}: {type(exc.reason).__name__}') from None
            time.sleep(2 ** attempt)


def cached_request(cache, name, path, params, *, refresh=False, insecure=False):
    cache.mkdir(parents=True, exist_ok=True)
    raw_path, meta_path = cache / (name + '.json'), cache / (name + '.meta.json')
    if raw_path.exists() or meta_path.exists():
        if not raw_path.exists() or not meta_path.exists():
            raise ValueError(f'Неполный кэш {name}; удалите повреждённую пару вручную или используйте --refresh')
        if not refresh:
            try:
                raw = raw_path.read_bytes(); meta = json.loads(meta_path.read_text())
                if hashlib.sha256(raw).hexdigest() != meta['sha256'] or meta['path'] != path or meta['params'] != [list(x) for x in params]:
                    raise ValueError
                return json.loads(raw), meta
            except (ValueError, KeyError, UnicodeError, json.JSONDecodeError):
                raise ValueError(f'Повреждён кэш {name}; используйте --refresh') from None
    raw, data = request_json(path, params, insecure=insecure)
    meta = dict(source=BASE, path=path, params=params,
                downloaded_at=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
                sha256=hashlib.sha256(raw).hexdigest(), schema_version=1)
    if refresh and (raw_path.exists() or meta_path.exists()):
        raise ValueError(f'Кэш {name} уже существует; --refresh требует нового каталога кэша, старые ответы сохранены')
    raw_path.write_bytes(raw)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return data, meta


def groups(data, field):
    if not isinstance(data, list): raise ValueError(f'Неверная схема {field}: ожидался список')
    result = {}
    for group in data:
        if not isinstance(group, dict) or not isinstance(group.get('m_Item2'), list):
            raise ValueError(f'Неверная схема {field}: группа')
        identifier = group.get('m_Item1')
        if identifier is None: continue
        if type(identifier) is not int or identifier in result:
            raise ValueError(f'Повтор или неизвестный ID в {field}')
        result[identifier] = group['m_Item2']
    return result


def indexed_hours(items, system, label, *, day=None):
    result = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f'Неверная строка {label} у {system}')
        hour = item.get('INTERVAL')
        if type(hour) is not int or not 0 <= hour <= 23:
            raise ValueError(f'Недопустимый час {label} у {system}: {hour}')
        if hour in result:
            raise ValueError(f'Дубли часов {label} у {system}: {hour}')
        if day is not None:
            stamp = item.get('M_DATE')
            try:
                parsed = datetime.fromisoformat(stamp) if isinstance(stamp, str) else None
            except ValueError:
                parsed = None
            if parsed is None or parsed.utcoffset() != timedelta(hours=3) or parsed.date() != day:
                raise ValueError(f'Неожиданная дата или зона {label} у {system}: {stamp}')
        result[hour] = item
    return result


def source_number(item, field, system, day, hour):
    if field not in item:
        raise ValueError(f'Нет поля {field} у {system} за {day}, час {hour}')
    value = item[field]
    if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
        raise ValueError(f'Некорректное число {field} у {system} за {day}, час {hour}')
    return value


def normalize(gen_data, price_days, start, end):
    generation = groups(gen_data, 'GenConsum')
    if set(generation) - set(TERRITORIES.values()): raise ValueError('Ответ содержит неизвестную территорию GenConsum')
    period_days = {start + timedelta(days=i) for i in range((end-start).days+1)}
    generation_by_system = {}
    for system in SYSTEMS:
        identifier = TERRITORIES[system]
        per_day = {}
        for item in generation.get(identifier, []):
            if not isinstance(item, dict): raise ValueError(f'Неверная строка GenConsum у {system}')
            stamp = item.get('M_DATE')
            try:
                parsed = datetime.fromisoformat(stamp) if isinstance(stamp, str) else None
            except ValueError:
                parsed = None
            if parsed is None or parsed.utcoffset() != timedelta(hours=3) or parsed.date() not in period_days:
                raise ValueError(f'Неожиданная дата или зона GenConsum у {system}: {stamp}')
            hour = item.get('INTERVAL')
            if type(hour) is not int or not 0 <= hour <= 23:
                raise ValueError(f'Недопустимый час GenConsum у {system}: {hour}')
            key = parsed.date(), hour
            if key in per_day: raise ValueError(f'Дубли часов GenConsum у {system}: {key}')
            per_day[key] = item
        generation_by_system[system] = per_day
    rows = []
    day = start
    while day <= end:
        if day.isoformat() not in price_days:
            raise ValueError(f'Не загружен ответ ИБР за {day}')
        prices = groups(price_days[day.isoformat()], 'GetHourlyData')
        if set(prices) - set(TERRITORIES.values()): raise ValueError('Ответ содержит неизвестную территорию GetHourlyData')
        for system in SYSTEMS:
            identifier = TERRITORIES[system]
            price_items = prices.get(identifier, [])
            price_by_hour = indexed_hours(price_items, system, 'GetHourlyData')
            for hour in range(24):
                g, p = generation_by_system[system].get((day, hour)), price_by_hour.get(hour)
                if g is None: continue
                price = source_number(p, 'AVERAGE_PRICE', system, day, hour) if p is not None else None
                row = dict(date=day.isoformat(), hour=hour, system=system, status='actual',
                    generation_mwh=source_number(g, 'GEN_FACT', system, day, hour),
                    consumption_mwh=source_number(g, 'E_USE_FACT', system, day, hour), ibr_rub_mwh=price)
                if row['generation_mwh'] is None: row['generation_missing_reason'] = 'Источник не опубликовал фактическую генерацию'
                if row['consumption_mwh'] is None: row['consumption_missing_reason'] = 'Источник не опубликовал фактическое потребление'
                if price is None: row['ibr_missing_reason'] = 'Источник не опубликовал ИБР' if p else 'Интервал ИБР не загружен'
                rows.append(row)
        day += timedelta(days=1)
    return rows


def fetch_month(start, end, cache, *, insecure=False, refresh=False, weight):
    ids = list(TERRITORIES.values())
    base = [('priceZone',''), *[('oesTerritory[]', str(x)) for x in ids]]
    params = base + [('startDate', start.strftime('%Y.%m.%d')), ('endDate', end.strftime('%Y.%m.%d'))]
    gen, gen_meta = cached_request(cache, 'genconsum', 'CommonInfo/GenConsum', params, refresh=refresh, insecure=insecure)
    prices, source_meta = {}, [gen_meta]
    day = start
    while day <= end:
        day_params = base + [('startDate', day.strftime('%Y.%m.%d'))]
        prices[day.isoformat()], meta = cached_request(cache, f'ibr_{day}', 'CommonInfo/GetHourlyData', day_params,
            refresh=refresh, insecure=insecure)
        source_meta.append(meta)
        day += timedelta(days=1)
    rows = normalize(gen, prices, start, end)
    snapshot = {'metadata': dict(source_url=BASE, downloaded_at=datetime.now(ZoneInfo('Europe/Moscow')).isoformat(),
        timezone='Europe/Moscow', hour_convention='interval_start_0_23', energy_unit='MWh', price_unit='RUB/MWh',
        weight_basis=weight, weight_explanation='Аналитическое допущение: объём фактического потребления; не официальная методика ИБР',
        month=start.strftime('%Y-%m'), schema_version=1, raw_cache=str(cache), raw_responses=[{'file': ('genconsum' if i==0 else f'ibr_{start+timedelta(days=i-1)}')+'.json', 'sha256': m['sha256']} for i,m in enumerate(source_meta)]), 'rows': rows}
    return snapshot


def main():
    parser = argparse.ArgumentParser(description='Загрузить реальные данные СБР и построить Excel')
    parser.add_argument('--month', type=parse_month)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--weight', choices=['consumption_mwh'], required=True, help='Явное аналитическое допущение для ИБР')
    parser.add_argument('--insecure-br', action='store_true', help='Обход TLS только для br.so-ups.ru без редиректов')
    parser.add_argument('--refresh', action='store_true', help='Загружать в новый каталог кэша')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Заменить готовый Excel, сохранив предыдущий набор в backup')
    args = parser.parse_args()
    start, end = args.month or previous_month()
    cache = args.cache or Path(__file__).parent/'data'/'raw'/start.strftime('%Y-%m')
    if args.refresh:
        cache = cache / ('refresh-' + datetime.now(ZoneInfo('Europe/Moscow')).strftime('%Y%m%dT%H%M%S%f'))
    snap = args.snapshot or Path(__file__).parent/'data'/'processed'/f'{start:%Y-%m}.json'
    output = args.output or Path(__file__).parent/'outputs'/f'electricity_{start:%Y-%m}.xlsx'
    try:
        if snap.exists() and not args.overwrite: raise ValueError(f'Снимок {snap} уже существует; выберите новый --snapshot или --overwrite')
        outputs = (output, output.with_suffix('.metadata.json'), output.with_suffix('.quality.json'))
        if any(path.exists() for path in outputs) and not args.overwrite:
            raise ValueError(f'Excel или его отчёты уже существуют: {output}; выберите новый --output или --overwrite')
        data = fetch_month(start, end, cache, insecure=args.insecure_br, refresh=args.refresh, weight=args.weight)
        validate(data['rows'], start, end, args.weight, allow_partial=args.allow_partial)
        snap.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.json', prefix='.snapshot-stage-', dir=snap.parent, delete=False) as staged:
            staged_path = Path(staged.name)
            json.dump(data, staged, ensure_ascii=False, indent=2, allow_nan=False)
        try:
            report = export(staged_path, output, start, end, args.weight, args.allow_partial, args.overwrite)
            if snap.exists():
                old = snap.with_name(snap.stem + '.backup-' + datetime.now(ZoneInfo('Europe/Moscow')).strftime('%Y%m%dT%H%M%S%f') + snap.suffix)
                snap.rename(old)
            staged_path.rename(snap)
        finally:
            staged_path.unlink(missing_ok=True)
        print(f'Создан {output}; наблюдений {report["observed_records"]}/{report["records"]}')
        return 0 if report['complete'] else 3
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Ошибка: {exc}\n')

if __name__ == '__main__': raise SystemExit(main())
