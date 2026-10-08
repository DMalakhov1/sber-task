"""Офлайн-проверка установки и артефактов. Не читает .env и не вызывает API."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def inspect(task='all'):
    tasks = {'brent': ('numpy', 'pandas', 'sklearn', 'statsmodels', 'matplotlib'),
             'rag': ('numpy', 'httpx', 'telegram', 'dotenv', 'pypdf', 'openai'),
             'excel': ('openpyxl',)}
    selected = tasks if task == 'all' else {task: tasks[task]}
    checks = []
    for name, packages in selected.items():
        missing = [p for p in packages if importlib.util.find_spec(p) is None]
        checks.append(dict(check=name+'_dependencies', ok=not missing, missing=missing))
    if 'brent' in selected:
        path = ROOT/'task1_brent/data/raw/brent_monthly_1987-2026.csv'
        checks.append(dict(check='brent_snapshot', ok=path.is_file()))
    if 'rag' in selected:
        path = ROOT/'task2_bot/rag_index/manifest.json'
        checks.append(dict(check='rag_index_present', ok=path.is_file(),
                           action='python -m task2_bot.rag.build_index (после подготовки и подтверждения PDF)'))
        checks.append(dict(check='semantic_package', ok=importlib.util.find_spec('sentence_transformers') is not None,
                           action='python -m pip install -r task2_bot/requirements-semantic.txt'))
    return {'network_used': False, 'secrets_read': False, 'checks': checks,
            'limits': ['Наличие пакета не подтверждает загрузку весов E5.',
                       'Параметры API, Telegram и фактическая выгрузка сайта здесь не проверяются.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['all','brent','rag','excel'],default='all')
    args=parser.parse_args()
    report=inspect(args.task)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if all(c['ok'] for c in report['checks']) else 2

if __name__=='__main__': raise SystemExit(main())
