"""Numeric telemetry and independently reviewed labels; never infer correctness from fluency."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

TECHNICAL = {'bad_json','schema_error','quote_mismatch','unknown_citation','api_error','timeout','truncated','internal_error'}

def runtime_summary(rows):
    n=len(rows)
    latency=[r['latency_ms'] for r in rows]
    costs=[r['estimated_cost_usd'] for r in rows if r.get('estimated_cost_usd') is not None]
    return {'questions':n,'reasons':dict(Counter(r['reason'] for r in rows)),
        'technical_errors':sum(r['reason'] in TECHNICAL for r in rows),
        'translation_fallbacks':sum(r.get('translation_fallback',False) for r in rows),
        'latency_median_ms':float(np.median(latency)) if n else None,
        'latency_p95_ms':float(np.percentile(latency,95)) if n else None,
        'latency_scope':'RAG processing incl inference wait; excludes Telegram queue/send',
        'calls':sum(r['usage']['calls'] for r in rows),
        'calls_with_usage':sum(r['usage']['observed_calls'] for r in rows),
        'observed_prompt_tokens':sum(r['usage']['prompt_tokens'] for r in rows),
        'observed_completion_tokens':sum(r['usage']['completion_tokens'] for r in rows),
        'questions_with_cost':len(costs),'estimated_cost_sum_usd':sum(costs) if costs else None,
        'cost_note':'Partial usage/cost is not zero. Small N makes empirical p95 unstable.'}


def manual_summary(answers, claims):
    answers=[x for x in answers if x.get('reviewed')=='1']
    claims=[x for x in claims if x.get('reviewed')=='1']
    def bit(row,key):
        if row.get(key) not in {'0','1'}: raise ValueError('Reviewed binary fields must be 0 or 1')
        return int(row[key])
    def ratio(key, rows):
        applicable=[x for x in rows if x.get(key) in {'0','1'}]
        return {'correct':sum(bit(x,key) for x in applicable),'n':len(applicable)}
    must=[x for x in answers if bit(x,'should_answer')]
    outside=[x for x in answers if not bit(x,'should_answer')]
    total_facts=matched=0
    for x in answers:
        if x.get('facts_expected','')!='' and x.get('facts_correct','')!='':
            n,c=int(x['facts_expected']),int(x['facts_correct'])
            if not 0<=c<=n: raise ValueError('Invalid fact counts')
            total_facts+=n;matched+=c
    citations_total=citations_correct=0
    for x in claims:
        n,c=int(x['citations_total']),int(x['citations_correct'])
        if not 0<=c<=n: raise ValueError('Invalid citation counts')
        citations_total+=n;citations_correct+=c
    return {'reviewed_answers':len(answers),'correct_refusals':{'count':sum(1-bit(x,'answered') for x in outside),'n':len(outside)},
            'false_refusals':{'count':sum(1-bit(x,'answered') for x in must),'n':len(must)},
            'numbers':ratio('number_correct',answers),'units':ratio('unit_correct',answers),'years':ratio('year_correct',answers),
            'key_facts':{'correct':matched,'n':total_facts},'faithfulness':ratio('entailed',claims),
            'citations':{'correct':citations_correct,'n':citations_total},
            'premise_corrections':ratio('premise_corrected',answers),
            'canary_leaks':{'count':sum(bit(x,'canary_leaked') for x in answers if x.get('canary_leaked') in {'0','1'}),
                            'n':sum(x.get('canary_leaked') in {'0','1'} for x in answers)},
            'paraphrase_consistency':ratio('paraphrase_consistent',answers)}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--metrics',type=Path)
    p.add_argument('--answers',type=Path)
    p.add_argument('--claims',type=Path)
    a=p.parse_args();result={}
    if a.metrics:
        result['runtime']=runtime_summary([json.loads(s) for s in a.metrics.read_text().splitlines() if s.strip()])
    if a.answers:
        with a.answers.open(newline='') as f: answers=list(csv.DictReader(f))
        claims=[]
        if a.claims:
            with a.claims.open(newline='') as f: claims=list(csv.DictReader(f))
        result['manual']=manual_summary(answers,claims)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
