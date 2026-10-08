"""Read-only audit of supplied auxiliary workbooks; not forecasting features."""
from datetime import date,datetime
from pathlib import Path
import re
import pandas as pd


def inventory(directory):
    from openpyxl import load_workbook
    records=[]
    for path in sorted(Path(directory).glob('*.xlsx')):
        book=load_workbook(path,read_only=True,data_only=True)
        rows=list(book.worksheets[0].values);book.close()
        col=1 if path.name=='usd_rub.xlsx' else 0
        dates=[]
        for row in rows[1:]:
            value=row[col] if len(row)>col else None
            if isinstance(value,(datetime,date)):dates.append(pd.Timestamp(value))
            elif isinstance(value,str) and re.fullmatch(r'\d{2}\.\d{4}',value):dates.append(pd.to_datetime(value,format='%m.%Y'))
        unique=sorted(set(dates));monthly=sorted(set(d.to_period('M') for d in dates))
        daily=path.name in ('ruonia.xlsx','usd_rub.xlsx')
        missing=[str(m) for m in pd.period_range(monthly[0],monthly[-1],freq='M') if m not in monthly] if monthly else []
        records.append(dict(file=path.name,rows=len(rows)-1,dated_rows=len(dates),first=str(min(dates).date()) if dates else '',
                            last=str(max(dates).date()) if dates else '',frequency='daily_business' if daily else 'monthly',
                            duplicate_dates=len(dates)-len(unique),extra_rows_in_month='' if daily else len(dates)-len(monthly),
                            missing_months=';'.join(missing),used_in_extended=False))
    return pd.DataFrame(records)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():p.exit(2,'Output already exists\n')
    inventory(Path(__file__).parent/'data/raw').to_csv(out,index=False)
