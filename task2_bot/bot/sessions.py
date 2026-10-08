from dataclasses import dataclass, field
import time

@dataclass(repr=False)
class Session:
    key: str = field(repr=False)
    touched: float
    last_request: float = float("-inf")
    last_trace: list = field(default_factory=list, repr=False)
    history: list = field(default_factory=list, repr=False)

class Sessions:
    def __init__(self, ttl=1800, capacity=100, clock=time.monotonic):
        self.ttl, self.capacity, self.clock = ttl, capacity, clock
        self._data = {}
        self._waiting = {}

    def purge(self):
        now = self.clock()
        for uid in list(self._data):
            if now - self._data[uid].touched >= self.ttl: self.reset(uid)
        for uid, until in list(self._waiting.items()):
            if now >= until: self._waiting.pop(uid, None)

    def reset(self, uid):
        session = self._data.pop(uid, None)
        if session:
            session.key = ""
            session.history.clear()
            session.last_trace.clear()
        self._waiting.pop(uid, None)

    def begin(self, uid):
        self.purge()
        self.reset(uid)
        if len(self._data) + len(self._waiting) >= self.capacity:
            raise ValueError("Слишком много сессий. Попробуйте позже.")
        self._waiting[uid] = self.clock() + 300

    def waiting(self, uid):
        self.purge()
        return uid in self._waiting

    def put(self, uid, key):
        self.purge()
        if uid not in self._waiting: raise ValueError("Сначала /start")
        self._waiting.pop(uid)
        self._data[uid] = Session(key=key, touched=self.clock())

    def get(self, uid):
        self.purge()
        session = self._data.get(uid)
        if session: session.touched = self.clock()
        return session

    def clear(self):
        for uid in list(self._data): self.reset(uid)
        self._waiting.clear()
