"""
ARC - Adaptive Replacement Cache
Megiddo & Modha, FAST 2003.

Maintains four LRU lists (OrderedDict, head = LRU, tail = MRU):
  T1  - pages seen exactly once recently (recency)
  T2  - pages seen at least twice (frequency)
  B1  - ghost / shadow for T1 (recently evicted from T1)
  B2  - ghost / shadow for T2 (recently evicted from T2)

An adaptive target p (in bytes) controls how much of the real cache
is reserved for T1.  Ghost hits push p toward T1 (B1 hit) or T2 (B2 hit).
"""

from collections import OrderedDict
from libcachesim import CommonCacheParams, Request


class ARCCache:
    def __init__(self, cache_size: int):
        self.c = cache_size          # total byte capacity
        self.p = 0                   # target bytes for T1 (adaptive)

        # Real cache lists (obj_id → size)
        self.T1: OrderedDict[int, int] = OrderedDict()
        self.T2: OrderedDict[int, int] = OrderedDict()
        # Ghost lists  (obj_id → size of the original object)
        self.B1: OrderedDict[int, int] = OrderedDict()
        self.B2: OrderedDict[int, int] = OrderedDict()

        self.T1_used = self.T2_used = 0
        self.B1_used = self.B2_used = 0

    # ── internal helpers ───────────────────────────────────────────────────────

    def _replace(self, hit_b2: bool) -> int:
        """
        Evict one item from T1 or T2 to B1 or B2.
        Returns the evicted object ID (now removed from the real cache).
        """
        # Decide: prefer T1 if it is above its target, or if a B2 hit is
        # happening and T1 is exactly at the target (tie-break toward T2).
        prefer_t1 = bool(self.T1) and (
            self.T1_used > self.p
            or (hit_b2 and self.T1_used == self.p)
        )

        if prefer_t1:
            victim_id, victim_size = next(iter(self.T1.items()))
            del self.T1[victim_id]
            self.T1_used -= victim_size
            self.B1[victim_id] = victim_size
            self.B1_used += victim_size
            self._trim_ghost(self.B1, "B1")
        elif self.T2:
            victim_id, victim_size = next(iter(self.T2.items()))
            del self.T2[victim_id]
            self.T2_used -= victim_size
            self.B2[victim_id] = victim_size
            self.B2_used += victim_size
            self._trim_ghost(self.B2, "B2")
        elif self.T1:
            # T2 empty fallback
            victim_id, victim_size = next(iter(self.T1.items()))
            del self.T1[victim_id]
            self.T1_used -= victim_size
            self.B1[victim_id] = victim_size
            self.B1_used += victim_size
            self._trim_ghost(self.B1, "B1")
        else:
            return 0

        return victim_id

    def _trim_ghost(self, ghost: OrderedDict, name: str) -> None:
        """Keep each ghost list from growing beyond c bytes."""
        attr = "B1_used" if name == "B1" else "B2_used"
        while getattr(self, attr) > self.c and ghost:
            _, old_size = ghost.popitem(last=False)
            setattr(self, attr, getattr(self, attr) - old_size)

    # ── hook implementations ───────────────────────────────────────────────────

    def on_hit(self, obj_id: int) -> None:
        """Promote from T1 → T2, or refresh position in T2."""
        if obj_id in self.T1:
            size = self.T1.pop(obj_id)
            self.T1_used -= size
            self.T2[obj_id] = size        # MRU of T2
            self.T2_used += size
        elif obj_id in self.T2:
            self.T2.move_to_end(obj_id)   # refresh to MRU

    def evict(self, new_id: int, new_size: int) -> int:
        """
        Called when the cache is full and new_id is about to be inserted.
        Adapts p, optionally trims a ghost list, then calls _replace.
        """
        in_b1 = new_id in self.B1
        in_b2 = new_id in self.B2

        if in_b1:
            # B1 hit → push p toward recency (T1 target grows)
            delta = max(new_size,
                        (self.B2_used * new_size) // max(1, self.B1_used))
            self.p = min(self.c, self.p + delta)
        elif in_b2:
            # B2 hit → push p toward frequency (T2 target grows)
            delta = max(new_size,
                        (self.B1_used * new_size) // max(1, self.B2_used))
            self.p = max(0, self.p - delta)
        else:
            # Completely new item: manage the directory size
            L1 = self.T1_used + self.B1_used
            L2 = self.T2_used + self.B2_used
            if L1 >= self.c and self.B1:
                old_id, old_size = self.B1.popitem(last=False)
                self.B1_used -= old_size
            elif L1 + L2 >= 2 * self.c and self.B2:
                old_id, old_size = self.B2.popitem(last=False)
                self.B2_used -= old_size

        return self._replace(in_b2)

    def on_miss(self, obj_id: int, obj_size: int) -> None:
        """
        Called after the core has inserted the new object.
        Places it in T1 or T2, removing from ghost if applicable.
        """
        if obj_id in self.B1:
            b1_size = self.B1.pop(obj_id)
            self.B1_used -= b1_size
            self.T2[obj_id] = obj_size
            self.T2_used += obj_size
        elif obj_id in self.B2:
            b2_size = self.B2.pop(obj_id)
            self.B2_used -= b2_size
            self.T2[obj_id] = obj_size
            self.T2_used += obj_size
        else:
            self.T1[obj_id] = obj_size
            self.T1_used += obj_size

    def on_remove(self, obj_id: int) -> None:
        """Explicit removal (not triggered by eviction)."""
        if obj_id in self.T1:
            self.T1_used -= self.T1.pop(obj_id)
        elif obj_id in self.T2:
            self.T2_used -= self.T2.pop(obj_id)


# ── libcachesim hook functions ──────────────────────────────────────────────


def init_hook(common_cache_params: CommonCacheParams):
    return ARCCache(common_cache_params.cache_size)


def hit_hook(data: ARCCache, req: Request) -> None:
    data.on_hit(req.obj_id)


def miss_hook(data: ARCCache, req: Request) -> None:
    data.on_miss(req.obj_id, req.obj_size)


def eviction_hook(data: ARCCache, req: Request) -> int:
    return data.evict(req.obj_id, req.obj_size)


def remove_hook(data: ARCCache, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: ARCCache) -> None:
    data.T1.clear(); data.T2.clear()
    data.B1.clear(); data.B2.clear()
    data.T1_used = data.T2_used = data.B1_used = data.B2_used = 0


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
        cache_name="arc",
    )

    trace = Path(__file__).parent.parent / "data" / "cloudPhysicsIO.vscsi"
    reader = TraceReader(trace=str(trace), trace_type=TraceType.VSCSI_TRACE)
    req_miss_ratio, byte_miss_ratio = cache.process_trace(reader)
    print(f"Request miss ratio: {req_miss_ratio:.4f}")
    print(f"Byte miss ratio:    {byte_miss_ratio:.4f}")
