from collections import deque
from libcachesim import CommonCacheParams, Request


class S3FifoCache:
    """
    S3-FIFO (Simple, Scalable, Sustainable FIFO) eviction policy.

    Three structures:
      S  – small FIFO queue  (~10 % of total cache bytes)
      M  – main  FIFO queue  (~90 % of total cache bytes), objects get one
           second chance (freq reset to 0 and re-enqueued) before eviction.
      G  – ghost set (IDs only, no data) of recently evicted-from-S objects,
           capped at the same byte budget as S.

    Admission:
      • Normal miss  → insert into S with freq = 0.
      • Ghost hit    → insert directly into M  (skip S).

    Eviction (called when the whole cache is full):
      1. Pop head of S.
         – If freq ≥ 1 → promote to M tail (freq reset to 0);
                          if M is now over budget, immediately evict from M.
         – If freq == 0 → evict; record in ghost G.
      2. If S is empty → evict from M.

    Evict from M: pop head; if freq ≥ 1 re-enqueue at tail with freq=0
    (one second chance); otherwise evict.
    """

    def __init__(self, cache_size: int):
        self.S_cap = max(1, cache_size // 10)
        self.M_cap = cache_size - self.S_cap

        # Queues: tuples (obj_id, size).  freq is tracked separately so we can
        # update it on hits without touching the queue.
        self.S: deque = deque()
        self.M: deque = deque()
        self.S_used = 0
        self.M_used = 0

        # Ghost: (obj_id, size) tuples so we can track byte budget correctly.
        self.ghost: deque = deque()
        self.ghost_set: set = set()
        self.ghost_used = 0

        self.freq: dict = {}      # obj_id → 0 | 1
        self.sizes: dict = {}     # obj_id → size  (needed for on_remove)
        self.location: dict = {}  # obj_id → 'S' | 'M'
        # Objects explicitly removed before they reach the head of a queue.
        # Stale entries are discarded lazily when they surface during eviction.
        self.removed: set = set()

    # ── helpers ────────────────────────────────────────────────────────────────

    def _ghost_trim(self) -> None:
        while self.ghost_used > self.S_cap and self.ghost:
            old_id, old_size = self.ghost.popleft()
            if old_id in self.ghost_set:
                self.ghost_set.discard(old_id)
                self.ghost_used -= old_size

    def _evict_from_M(self) -> int:
        while self.M:
            obj_id, size = self.M.popleft()
            self.M_used -= size

            if obj_id in self.removed:
                # Already gone from the real cache; just clean up bookkeeping.
                self.removed.discard(obj_id)
                self.freq.pop(obj_id, None)
                self.sizes.pop(obj_id, None)
                self.location.pop(obj_id, None)
                continue

            if self.freq.get(obj_id, 0) > 0:
                # Second chance: reset freq and re-enqueue at tail.
                self.freq[obj_id] = 0
                self.M.append((obj_id, size))
                self.M_used += size
            else:
                self.freq.pop(obj_id, None)
                self.sizes.pop(obj_id, None)
                self.location.pop(obj_id, None)
                return obj_id

        return 0  # should not happen if cache is not empty

    # ── Policy hooks ───────────────────────────────────────────────────────────

    def on_miss(self, obj_id: int, obj_size: int) -> None:
        self.sizes[obj_id] = obj_size
        self.freq[obj_id] = 0

        if obj_id in self.ghost_set:
            # Ghost hit: skip S, insert directly into M.
            self.ghost_set.discard(obj_id)
            self.location[obj_id] = "M"
            self.M.append((obj_id, obj_size))
            self.M_used += obj_size
        else:
            self.location[obj_id] = "S"
            self.S.append((obj_id, obj_size))
            self.S_used += obj_size

    def on_hit(self, obj_id: int) -> None:
        if obj_id in self.freq:
            self.freq[obj_id] = min(1, self.freq[obj_id] + 1)

    def evict(self) -> int:
        while self.S:
            obj_id, size = self.S.popleft()
            self.S_used -= size

            if obj_id in self.removed:
                self.removed.discard(obj_id)
                self.freq.pop(obj_id, None)
                self.sizes.pop(obj_id, None)
                self.location.pop(obj_id, None)
                continue

            if self.freq.get(obj_id, 0) > 0:
                # Promote to M.
                self.freq[obj_id] = 0
                self.location[obj_id] = "M"
                self.M.append((obj_id, size))
                self.M_used += size
                if self.M_used > self.M_cap:
                    return self._evict_from_M()
                # M still has room; keep scanning S for a freq-0 victim.
            else:
                # Evict; record in ghost.
                self.freq.pop(obj_id, None)
                self.sizes.pop(obj_id, None)
                self.location.pop(obj_id, None)
                self.ghost_set.add(obj_id)
                self.ghost.append((obj_id, size))
                self.ghost_used += size
                self._ghost_trim()
                return obj_id

        # S is empty; fall back to M.
        return self._evict_from_M()

    def on_remove(self, obj_id: int) -> None:
        if obj_id not in self.location:
            return
        # Mark as removed; bookkeeping (used counters) will be corrected lazily
        # when the stale entry is popped from its queue during the next eviction.
        self.location.pop(obj_id)
        self.freq.pop(obj_id, None)
        self.removed.add(obj_id)
        # Keep sizes[obj_id] so we know the size when the stale entry surfaces.


# ── libcachesim hook functions ──────────────────────────────────────────────


def init_hook(common_cache_params: CommonCacheParams):
    return S3FifoCache(common_cache_params.cache_size)


def hit_hook(data: S3FifoCache, req: Request) -> None:
    data.on_hit(req.obj_id)


def miss_hook(data: S3FifoCache, req: Request) -> None:
    data.on_miss(req.obj_id, req.obj_size)


def eviction_hook(data: S3FifoCache, req: Request) -> int:
    return data.evict()


def remove_hook(data: S3FifoCache, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: S3FifoCache) -> None:
    data.S.clear()
    data.M.clear()
    data.ghost.clear()
    data.ghost_set.clear()
    data.freq.clear()
    data.sizes.clear()
    data.location.clear()
    data.removed.clear()


# ── local smoke-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    from libcachesim import PluginCache, TraceReader, TraceType

    cache = PluginCache(
        cache_size=1024 * 1024,
        cache_init_hook=init_hook,
        cache_hit_hook=hit_hook,
        cache_miss_hook=miss_hook,
        cache_eviction_hook=eviction_hook,
        cache_remove_hook=remove_hook,
        cache_free_hook=free_hook,
        cache_name="s3fifo",
    )

    trace = Path(__file__).parent.parent / "data" / "cloudPhysicsIO.vscsi"
    reader = TraceReader(trace=str(trace), trace_type=TraceType.VSCSI_TRACE)
    req_miss_ratio, byte_miss_ratio = cache.process_trace(reader)
    print(f"Request miss ratio: {req_miss_ratio:.4f}")
    print(f"Byte miss ratio:    {byte_miss_ratio:.4f}")
