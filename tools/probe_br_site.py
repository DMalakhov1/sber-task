"""Bounded reconnaissance of known URLs, not a working data adapter."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

URLS = ('https://br.so-ups.ru/', 'https://br.so-ups.ru/robots.txt',
        'http://br.so-ups.ru:8090/PublicApi/PublicApiService.svc?wsdl')
LIMIT = 1024 * 1024

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Record status; do not follow an unknown destination.


def wsdl_summary(body):
    if b'<!DOCTYPE' in body.upper() or b'<!ENTITY' in body.upper():
        raise ValueError('DTD/entities are not accepted')
    root = ET.fromstring(body)
    if root.tag != '{http://schemas.xmlsoap.org/wsdl/}definitions':
        raise ValueError('Not WSDL 1.1')
    ns = {'w': 'http://schemas.xmlsoap.org/wsdl/'}
    return {'operations': sorted({x.attrib['name'] for x in root.findall('.//w:portType/w:operation', ns)}),
            'imports': [x.get('location') for x in root.findall('w:import', ns)],
            'note': 'Imports not fetched; operations do not establish data semantics.'}


def probe(url, timeout, output):
    started = time.monotonic()
    result = {'url': url}
    opener = urllib.request.build_opener(NoRedirect())
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'SberTestRecon/1.0'})
        with opener.open(req, timeout=timeout) as response:
            result.update(status=response.status, content_type=response.headers.get('Content-Type', ''))
            body = response.read(LIMIT + 1)
            result['truncated'] = len(body) > LIMIT
            if url.endswith('?wsdl') and len(body) <= LIMIT:
                try:
                    result['wsdl'] = wsdl_summary(body)
                    result['sha256'] = hashlib.sha256(body).hexdigest()
                    (output/'service.wsdl').write_bytes(body)
                except (ET.ParseError, ValueError, KeyError):
                    result['wsdl_error'] = 'Response is not a supported WSDL'
            # HTML and response headers are deliberately not persisted (hidden tokens).
    except urllib.error.HTTPError as exc:
        result.update(status=exc.code, error='HTTPError')
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        result['error'] = type(exc).__name__
        result['cause_type'] = type(getattr(exc, 'reason', exc)).__name__
    result['elapsed_seconds'] = round(time.monotonic()-started, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=float, default=10)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 0 < args.timeout <= 30:
        parser.error('timeout must be in (0, 30] seconds')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or Path('docs/site_recon')/stamp
    output.mkdir(parents=True, exist_ok=False)
    report = {'checked_at_utc': stamp, 'historical_api_unconfirmed': True,
              'results': [probe(url, args.timeout, output) for url in URLS]}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if any('wsdl' in r for r in report['results']) else 2

if __name__ == '__main__':
    raise SystemExit(main())
