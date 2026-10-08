"""Save explicit Excel formulas with independently calculated numeric caches."""
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED

NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'


class CachedWorkbook:
    def __init__(self, workbook, cache):
        self.workbook, self.cache = workbook, cache

    def save(self, path):
        self.workbook.save(path)
        path = Path(path)
        with ZipFile(path) as z:
            payload = {name: z.read(name) for name in z.namelist()}
        for index, sheet in enumerate(self.workbook.worksheets, 1):
            member = f'xl/worksheets/sheet{index}.xml'
            root = ET.fromstring(payload[member])
            for cell in root.iter(f'{{{NS}}}c'):
                key = (sheet.title, cell.attrib['r'])
                if key not in self.cache: continue
                if cell.find(f'{{{NS}}}f') is None: raise ValueError('Cache without formula')
                value = self.cache[key]
                cached = cell.find(f'{{{NS}}}v')
                if cached is None: cached = ET.SubElement(cell, f'{{{NS}}}v')
                if value is None:
                    cell.set('t', 'str'); cached.text = None
                else:
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError('Invalid formula cache')
                    cell.set('t', 'n'); cached.text = repr(value)
            payload[member] = ET.tostring(root, encoding='utf-8', xml_declaration=True)
        with ZipFile(path, 'w', ZIP_DEFLATED) as z:
            for name, data in payload.items(): z.writestr(name, data)


def strict_sum(values, size):
    return f'=IF(COUNT({values})={size},SUM({values}),"")'


def strict_weighted(prices, weights, size):
    return (f'=IF(AND(COUNT({prices})={size},COUNT({weights})={size},SUM({weights})>0),'
            f'SUMPRODUCT({prices},{weights})/SUM({weights}),"")')


def available_formulas(prices, weights, size):
    numerator = f'SUMPRODUCT({prices},{weights})'
    denominator = f'SUMIF({prices},"<>",{weights})'
    return [f'=IF({denominator}>0,{numerator}/{denominator},"")',
            f'=IF(COUNT({prices})>0,AVERAGE({prices}),"")',
            f'=COUNT({prices})',
            f'=IF(AND(COUNT({weights})={size},SUM({weights})>0),{denominator}/SUM({weights}),"")']
