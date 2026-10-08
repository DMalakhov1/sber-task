import asyncio
from contextlib import asynccontextmanager

class UserLocks:
    """One user is sequential; independent users can proceed concurrently."""
    def __init__(self): self._entries = {}

    @asynccontextmanager
    async def hold(self, user_id):
        # No await during refcount changes: atomic on this single event loop.
        entry = self._entries.setdefault(user_id, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]: yield
        finally:
            entry[1] -= 1
            if entry[1] == 0: self._entries.pop(user_id, None)
