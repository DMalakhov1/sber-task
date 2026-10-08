"""Искусственный пример оформления Excel. НЕ фактические данные сайта."""
from datetime import date
import json
from pathlib import Path
import tempfile
from openpyxl import load_workbook
from .aggregation import SYSTEMS
from .export_excel import export


def main():
    output=Path(__file__).parent/'outputs'/'DEMO_SYNTHETIC.xlsx'
    if output.exists():
        print(f'Синтетический пример уже существует: {output}')
        return
    data={'metadata':dict(source_url='urn:synthetic:task3-demo',downloaded_at='not_applicable_synthetic',
          timezone='Europe/Moscow',hour_convention='interval_start_0_23',energy_unit='MWh',price_unit='RUB/MWh',
          weight_basis='consumption_mwh',weight_explanation='СИНТЕТИЧЕСКИЙ ПРИМЕР: вес потребления выбран только для демонстрации.',synthetic=True),
          'rows':[dict(date=f'2026-09-{d:02}',hour=h,system=s,status='actual',generation_mwh=100+i*10+h,
                       consumption_mwh=90+i*10+h,ibr_rub_mwh=1000+i*100+h*10)
                  for d in range(1,31) for h in range(24) for i,s in enumerate(SYSTEMS)]}
    # status=actual exercises the contract only; every sheet is visibly marked synthetic.
    with tempfile.TemporaryDirectory() as tmp:
        snapshot=Path(tmp)/'synthetic.json';snapshot.write_text(json.dumps(data,ensure_ascii=False))
        export(snapshot,output,date(2026,9,1),date(2026,9,30),'consumption_mwh')
    print(f'Создан синтетический пример: {output}')


if __name__=='__main__': main()
