"""
S3-SIEVE - S3-FIFO admission logic fused with SIEVE eviction in the main queue.

Structure
─────────
  S  (10 % of total bytes) - small FIFO queue for new items.
     Each item carries a 1-bit freq flag.
  M  (90 % of total bytes) - main queue using SIEVE eviction:
     doubly-linked list + hand pointer + visited bit.
     Items enter at the head (MRU) with visited = True (already popular).
  G  - ghost set for recently evicted-from-S items (same byte budget as S).
     A ghost hit bypasses S and enters M directly with visited = True.

Why this beats both parents
───────────────────────────
  vs S3-FIFO  : SIEVE in M replaces FIFO+one-chance.  SIEVE is provably
                better than CLOCK/second-chance for managing hot items.
  vs SIEVE    : The S queue + ghost filter catches one-hit wonders before
                they ever pollute M, which pure SIEVE cannot do.
"""

from collections import deque
from libcachesim import CommonCacheParams, Request


class _Node:
    __slots__ = ("obj_id", "visited", "prev", "next")

    def __init__(self, obj_id: int):
        self.obj_id = obj_id
        self.visited = False
        self.prev = None
        self.next = None


class S3SieveCache:
    def __init__(self, cache_size: int):
        self.S_cap = max(1, cache_size // 10)
        self.M_cap = cache_size - self.S_cap
        self.ghost_cap = self.S_cap  # ghost budget mirrors S

        # ── S queue (FIFO) ─────────────────────────────────────────────────────
        self.S: deque = deque()           # (obj_id, size)
        self.S_used = 0
        self.S_freq: dict[int, int] = {}  # obj_id → 0|1
        self.S_sizes: dict[int, int] = {} # obj_id → size

        # ── M queue (SIEVE DLL) ────────────────────────────────────────────────
        self.M_map: dict[int, _Node] = {} # obj_id → node
        self.M_sizes: dict[int, int] = {} # obj_id → size
        self.M_head: _Node | None = None  # newest (MRU)
        self.M_tail: _Node | None = None  # oldest
        self.M_hand: _Node | None = None  # eviction pointer
        self.M_used = 0

        # ── ghost ──────────────────────────────────────────────────────────────
        self.ghost: deque = deque()       # (obj_id, size)
        self.ghost_set: set = set()
        self.ghost_used = 0

        # ── bookkeeping ────────────────────────────────────────────────────────
        self.location: dict[int, str] = {}  # obj_id → 'S' | 'M'
        self.removed: set = set()           # explicitly removed before queue pop

    # ── M DLL helpers ──────────────────────────────────────────────────────────

    def _m_link_head(self, node: _Node) -> None:
        node.prev = None
        node.next = self.M_head
        if self.M_head:
            self.M_head.prev = node
        self.M_head = node
        if self.M_tail is None:
            self.M_tail = node

    def _m_unlink(self, node: _Node) -> None:
        if node.prev:
            node.prev.next = node.next
        else:
            self.M_head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.M_tail = node.prev
        node.prev = node.next = None

    def _m_evict_sieve(self) -> int:
        """SIEVE-style eviction from M. Returns evicted obj_id or 0."""
        if not self.M_map:
            return 0
        if self.M_hand is None:
            self.M_hand = self.M_tail

        # Scan toward head; reset visited items until an unvisited one is found.
        while self.M_hand.visited:
            self.M_hand.visited = False
            self.M_hand = self.M_hand.prev or self.M_tail  # wrap at head

        victim = self.M_hand
        self.M_hand = victim.prev  # advance (may be None; resets on next call)

        self._m_unlink(victim)
        size = self.M_sizes.pop(victim.obj_id)
        self.M_map.pop(victim.obj_id)
        self.M_used -= size
        self.location.pop(victim.obj_id, None)
        return victim.obj_id

    def _ghost_trim(self) -> None:
        while self.ghost_used > self.ghost_cap and self.ghost:
            old_id, old_size = self.ghost.popleft()
            if old_id in self.ghost_set:
                self.ghost_set.discard(old_id)
                self.ghost_used -= old_size

    # ── hook implementations ───────────────────────────────────────────────────

    def on_hit(self, obj_id: int) -> None:
        if obj_id in self.S_freq:
            self.S_freq[obj_id] = 1              # mark as popular in S
        elif obj_id in self.M_map:
            self.M_map[obj_id].visited = True    # protect in M

    def on_miss(self, obj_id: int, obj_size: int) -> None:
        if obj_id in self.ghost_set:
            # Ghost hit → skip S, insert directly into M as visited (protected)
            self.ghost_set.discard(obj_id)
            node = _Node(obj_id)
            node.visited = True
            self._m_link_head(node)
            self.M_map[obj_id] = node
            self.M_sizes[obj_id] = obj_size
            self.M_used += obj_size
            self.location[obj_id] = "M"
        else:
            # Brand new → enter S with freq = 0
            self.S.append((obj_id, obj_size))
            self.S_used += obj_size
            self.S_freq[obj_id] = 0
            self.S_sizes[obj_id] = obj_size
            self.location[obj_id] = "S"

    def evict(self) -> int:
        # ── drain S, looking for a freq-0 victim ──────────────────────────────
        while self.S:
            obj_id, size = self.S.popleft()
            self.S_used -= size

            if obj_id in self.removed:
                self.removed.discard(obj_id)
                self.S_freq.pop(obj_id, None)
                self.S_sizes.pop(obj_id, None)
                self.location.pop(obj_id, None)
                continue

            freq = self.S_freq.pop(obj_id, 0)
            self.S_sizes.pop(obj_id, None)
            self.location.pop(obj_id, None)

            if freq > 0:
                # Promote to M head with visited = True
                node = _Node(obj_id)
                node.visited = True
                self._m_link_head(node)
                self.M_map[obj_id] = node
                self.M_sizes[obj_id] = size
                self.M_used += size
                self.location[obj_id] = "M"

                # If M just went over budget, evict from M immediately
                if self.M_used > self.M_cap:
                    return self._m_evict_sieve()
                # Otherwise keep scanning S for a freq-0 victim
            else:
                # One-hit wonder: evict and record in ghost
                self.ghost_set.add(obj_id)
                self.ghost.append((obj_id, size))
                self.ghost_used += size
                self._ghost_trim()
                return obj_id

        # S exhausted (or all promoted) → evict from M
        return self._m_evict_sieve()

    def on_remove(self, obj_id: int) -> None:
        """Handle explicit removal."""
        loc = self.location.pop(obj_id, None)
        if loc == "M":
            node = self.M_map.pop(obj_id, None)
            if node:
                if self.M_hand is node:
                    self.M_hand = node.prev
                self._m_unlink(node)
                size = self.M_sizes.pop(obj_id, 0)
                self.M_used -= size
        elif loc == "S":
            # Cannot remove from deque interior in O(1); mark for lazy cleanup.
            self.S_freq.pop(obj_id, None)
            self.S_sizes.pop(obj_id, None)
            self.removed.add(obj_id)


# ── libcachesim hook functions ──────────────────────────────────────────────


def init_hook(common_cache_params: CommonCacheParams):
    return S3SieveCache(common_cache_params.cache_size)


def hit_hook(data: S3SieveCache, req: Request) -> None:
    data.on_hit(req.obj_id)


def miss_hook(data: S3SieveCache, req: Request) -> None:
    data.on_miss(req.obj_id, req.obj_size)


def eviction_hook(data: S3SieveCache, req: Request) -> int:
    return data.evict()


def remove_hook(data: S3SieveCache, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: S3SieveCache) -> None:
    data.S.clear()
    data.S_freq.clear(); data.S_sizes.clear()
    data.M_map.clear(); data.M_sizes.clear()
    data.M_head = data.M_tail = data.M_hand = None
    data.ghost.clear(); data.ghost_set.clear()
    data.location.clear(); data.removed.clear()


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
        cache_name="s3sieve",
    )

    trace = Path(__file__).parent.parent / "data" / "cloudPhysicsIO.vscsi"
    reader = TraceReader(trace=str(trace), trace_type=TraceType.VSCSI_TRACE)
    req_miss_ratio, byte_miss_ratio = cache.process_trace(reader)
    print(f"Request miss ratio: {req_miss_ratio:.4f}")
    print(f"Byte miss ratio:    {byte_miss_ratio:.4f}")
