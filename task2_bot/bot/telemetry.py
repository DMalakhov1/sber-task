"""Per-request counters. Never serialize user input, identity, credentials or exception text."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import threading

REASONS = {'ok','abstain','bad_json','schema_error','quote_mismatch','unknown_citation','no_hits',
           'api_error','timeout','truncated','translation_invalid','input_invalid','internal_error','verification_invalid','unsupported_claim','off_topic_answer'}
CURRENT_USAGE = ContextVar('rag_usage', default=None)

@dataclass
class Usage:
    calls: int = 0
    observed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    cache_known_calls: int = 0

    def add(self, raw):
        def integer(x): return isinstance(x, int) and not isinstance(x, bool) and x >= 0
        if not isinstance(raw, dict): return
        prompt, completion = raw.get('prompt_tokens'), raw.get('completion_tokens')
        if not integer(prompt) or not integer(completion): return
        self.observed_calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        hit, miss = raw.get('prompt_cache_hit_tokens'), raw.get('prompt_cache_miss_tokens')
        if hit is None and isinstance(raw.get('prompt_tokens_details'), dict):
            hit = raw['prompt_tokens_details'].get('cached_tokens')
        if integer(hit) and miss is None: miss = prompt - hit
        if integer(hit) and integer(miss) and hit + miss == prompt:
            self.cache_known_calls += 1
            self.cache_hit_tokens += hit
            self.cache_miss_tokens += miss

    def estimated_cost(self, prices):
        # No tariff defaults: price depends on model/date/time tier. Missing data is NOT zero cost.
        if self.calls == 0: return 0.0
        if self.observed_calls != self.calls or self.cache_known_calls != self.calls or not prices: return None
        keys = ('input_hit_per_million','input_miss_per_million','output_per_million')
        if any(not isinstance(prices.get(k), (int,float)) or isinstance(prices[k], bool)
               or not math.isfinite(prices[k]) or prices[k] < 0 for k in keys): return None
        return (self.cache_hit_tokens * prices[keys[0]] + self.cache_miss_tokens * prices[keys[1]]
                + self.completion_tokens * prices[keys[2]]) / 1_000_000

@contextmanager
def collect_usage():
    usage = Usage()
    token = CURRENT_USAGE.set(usage)
    try: yield usage
    finally: CURRENT_USAGE.reset(token)

class MetricSink:
    def __init__(self, path, prices=None):
        self.path, self.prices = Path(path), prices
        self._lock = threading.Lock()

    def write(self, result):
        # Whitelist, not a dump of result.__dict__. No user ID, question hash or text, no timestamp.
        row = {'reason': result.reason if result.reason in REASONS else 'internal_error',
               'latency_ms': round(result.latency_ms, 2),
               'rerank_status': result.rerank_status if result.rerank_status in {'off','ok','fallback','no_hits'} else 'off',
               'translation_fallback': bool(result.translation_fallback),
               'translation_reason': result.translation_reason if result.translation_reason in REASONS else None,
               'repair_attempted': bool(result.repair_attempted),
               'initial_reason': result.initial_reason if result.initial_reason in REASONS else None,
               'retrieved_count': len(result.trace), 'usage': asdict(result.usage),
               'estimated_cost_usd': result.usage.estimated_cost(self.prices)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
