"""
W-TinyLFU - Window TinyLFU cache replacement policy.
Based on: "TinyLFU: A Highly Efficient Cache Admission Filter" (Ben-David et al.)
and the Caffeine implementation (Karp & Meisel).

Architecture
------------
  Window  (1% of total bytes)   - pure LRU; absorbs new/bursty items.
  Main    (99% of total bytes)  - recently re-accessed items, rarely evicted.
      Probation  (20% of main)  - demoted from Protected or admitted from Window.

Admission filter
----------------
When the Window LRU overflows, the tail item (candidate) competes with the
tail of Probation (victim).  The one with *higher* estimated frequency stays;
the other is evicted from the cache.

Frequency sketch
----------------
A Count-Min Sketch (4 rows * power-of-2 width) approximates per-object
access frequency.  After every `reset_threshold` increments all counters are
halved (aging), so stale frequency decays over time.
"""

from collections import OrderedDict
from libcachesim import CommonCacheParams, Request


class _CountMinSketch:
    """4-row Count-Min Sketch with periodic halving for recency decay."""

    DEPTH = 4
    # Large multipliers spread hashes well across the table.
    _SEEDS = [0xABC9_7531, 0xDEF0_1357, 0x2468_ACE0, 0x1357_9BDF]

    def __init__(self, width: int):
        # Width must be a power of two for fast modulo via bit-mask.
        w = 1
        while w < max(8, width):
            w <<= 1
        self.width = w
        self.mask = w - 1
        self.table = [[0] * w for _ in range(self.DEPTH)]
        self.size = 0
        self.reset_threshold = w * 8   # halve counters after this many adds

    def increment(self, key: int) -> None:
        for d, seed in enumerate(self._SEEDS):
            idx = (key * seed) & self.mask
            v = self.table[d][idx]
            if v < 15:                   # cap at 15 to fit in a nibble
                self.table[d][idx] = v + 1
        self.size += 1
        if self.size >= self.reset_threshold:
            self._reset()

    def estimate(self, key: int) -> int:
        return min(self.table[d][(key * seed) & self.mask]
                   for d, seed in enumerate(self._SEEDS))

    def _reset(self) -> None:
        """Halve all counters (divide by 2) to age out stale frequencies."""
        for row in self.table:
            for i in range(self.width):
                row[i] >>= 1
        self.size >>= 1


class WTinyLFUCache:
    def __init__(self, cache_size: int):
        self.cache_size = cache_size

        # Segment byte budgets
        self.win_cap  = max(1, cache_size // 100)          # ~1 %
        self.main_cap = cache_size - self.win_cap
        self.prot_cap = max(1, int(self.main_cap * 0.8))   # 80 % of main

        # LRU segments: head = LRU (eviction end), tail = MRU (insertion end)
        self.win:  OrderedDict[int, int] = OrderedDict()   # obj_id → size
        self.prot: OrderedDict[int, int] = OrderedDict()
        self.prob: OrderedDict[int, int] = OrderedDict()

        self.win_used  = 0
        self.prot_used = 0
        self.prob_used = 0

        # Sketch width: ~8× number of expected distinct objects
        # We estimate ~(cache_size / 512) objects as a rough default.
        sketch_width = max(256, cache_size // 512)
        self.sketch = _CountMinSketch(sketch_width)

        self.removed: set = set()   # explicitly removed objects

    # ── helpers ────────────────────────────────────────────────────────────────

    def _promote_to_protected(self, obj_id: int, size: int) -> None:
        """Move an item from Probation to the MRU end of Protected."""
        del self.prob[obj_id]
        self.prob_used -= size
        self.prot[obj_id] = size
        self.prot_used += size
        # Demote Protected LRU → Probation if over budget
        while self.prot_used > self.prot_cap and self.prot:
            old_id, old_size = self.prot.popitem(last=False)
            self.prot_used -= old_size
            if old_id not in self.removed:
                self.prob[old_id] = old_size
                self.prob_used += old_size
            else:
                self.removed.discard(old_id)

    def _next_in(self, d: OrderedDict) -> tuple[int, int] | None:
        """Peek at the LRU end of an OrderedDict, skipping removed items."""
        while d:
            obj_id, size = next(iter(d.items()))
            if obj_id in self.removed:
                del d[obj_id]
                self.removed.discard(obj_id)
                # Adjust used counters lazily (approximate; see on_remove)
                continue
            return obj_id, size
        return None

    # ── hook implementations ───────────────────────────────────────────────────

    def on_hit(self, obj_id: int) -> None:
        self.sketch.increment(obj_id)
        if obj_id in self.win:
            self.win.move_to_end(obj_id)
        elif obj_id in self.prob:
            size = self.prob[obj_id]
            self._promote_to_protected(obj_id, size)
        elif obj_id in self.prot:
            self.prot.move_to_end(obj_id)

    def on_miss(self, obj_id: int, obj_size: int) -> None:
        self.sketch.increment(obj_id)
        # New items always enter the Window at the MRU end.
        self.win[obj_id] = obj_size
        self.win_used += obj_size

    def evict(self) -> int:
        """
        Choose one object to evict.  Maintains the W-TinyLFU invariant:
          1. If the Window LRU (candidate) has lower frequency than the
             Probation LRU (victim), evict the candidate (admission rejected).
          2. Otherwise admit candidate to Probation and evict the victim.
          3. If no Probation items exist, fall through to Probation → Protected.
        """
        # ── step 1: process the Window LRU candidate ──────────────────────────
        win_item = self._next_in(self.win)
        if win_item:
            cand_id, cand_size = win_item
            prob_item = self._next_in(self.prob)

            if prob_item:
                vict_id, vict_size = prob_item
                if self.sketch.estimate(cand_id) < self.sketch.estimate(vict_id):
                    # Candidate loses → evict it (reject from main)
                    del self.win[cand_id]
                    self.win_used -= cand_size
                    return cand_id
                else:
                    # Candidate wins → admit to Probation, evict the victim
                    del self.win[cand_id]
                    self.win_used -= cand_size
                    self.prob[cand_id] = cand_size
                    self.prob_used += cand_size

                    del self.prob[vict_id]
                    self.prob_used -= vict_size
                    return vict_id

            else:
                # Probation is empty; move candidate there and evict from Protected
                del self.win[cand_id]
                self.win_used -= cand_size
                self.prob[cand_id] = cand_size
                self.prob_used += cand_size

                prot_item = self._next_in(self.prot)
                if prot_item:
                    vict_id, vict_size = prot_item
                    del self.prot[vict_id]
                    self.prot_used -= vict_size
                    return vict_id
                # Protected also empty – evict the candidate we just admitted
                del self.prob[cand_id]
                self.prob_used -= cand_size
                return cand_id

        # ── step 2: Window is empty; evict from Probation ─────────────────────
        prob_item = self._next_in(self.prob)
        if prob_item:
            vict_id, vict_size = prob_item
            del self.prob[vict_id]
            self.prob_used -= vict_size
            return vict_id

        # ── step 3: fall back to Protected ────────────────────────────────────
        prot_item = self._next_in(self.prot)
        if prot_item:
            vict_id, vict_size = prot_item
            del self.prot[vict_id]
            self.prot_used -= vict_size
            return vict_id

        return 0

    def on_remove(self, obj_id: int) -> None:
        """
        Mark as removed; stale entries are cleaned up lazily in _next_in.
        We eagerly pop from the dict if it happens to be in one of the segments,
        because that keeps used-counters accurate without a full scan.
        """
        for seg, attr in ((self.win,  "win_used"),
                          (self.prob, "prob_used"),
                          (self.prot, "prot_used")):
            if obj_id in seg:
                setattr(self, attr, getattr(self, attr) - seg.pop(obj_id))
                return
        self.removed.add(obj_id)


# ── libcachesim hook functions ──────────────────────────────────────────────


def init_hook(common_cache_params: CommonCacheParams):
    return WTinyLFUCache(common_cache_params.cache_size)


def hit_hook(data: WTinyLFUCache, req: Request) -> None:
    data.on_hit(req.obj_id)


def miss_hook(data: WTinyLFUCache, req: Request) -> None:
    data.on_miss(req.obj_id, req.obj_size)


def eviction_hook(data: WTinyLFUCache, req: Request) -> int:
    return data.evict()


def remove_hook(data: WTinyLFUCache, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: WTinyLFUCache) -> None:
    data.win.clear(); data.prot.clear(); data.prob.clear()
    data.sketch.table = [[0] * data.sketch.width for _ in range(data.sketch.DEPTH)]
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
        cache_name="wtinylfu",
    )

    trace = Path(__file__).parent.parent / "data" / "cloudPhysicsIO.vscsi"
    reader = TraceReader(trace=str(trace), trace_type=TraceType.VSCSI_TRACE)
    req_miss_ratio, byte_miss_ratio = cache.process_trace(reader)
    print(f"Request miss ratio: {req_miss_ratio:.4f}")
    print(f"Byte miss ratio:    {byte_miss_ratio:.4f}")
