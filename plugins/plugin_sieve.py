from libcachesim import CommonCacheParams, Request


class _Node:
    __slots__ = ("obj_id", "visited", "prev", "next")

    def __init__(self, obj_id: int):
        self.obj_id = obj_id
        self.visited = False
        self.prev = None
        self.next = None


class SieveCache:
    """
    SIEVE eviction policy.

    New objects are inserted at the head of a doubly-linked list.
    A 'hand' pointer scans from the tail toward the head.
    - visited == True  → reset to False, advance hand (second chance).
    - visited == False → evict this object.
    The hand wraps back to the tail when it reaches the head.
    """

    def __init__(self, cache_size: int):
        self.cache_size = cache_size
        self.nodes: dict[int, _Node] = {}
        self.head: _Node | None = None  # newest
        self.tail: _Node | None = None  # oldest
        self.hand: _Node | None = None  # eviction pointer

    # ── DLL helpers ────────────────────────────────────────────────────────────

    def _link_at_head(self, node: _Node) -> None:
        node.prev = None
        node.next = self.head
        if self.head:
            self.head.prev = node
        self.head = node
        if self.tail is None:
            self.tail = node

    def _unlink(self, node: _Node) -> None:
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        node.prev = node.next = None

    # ── Policy hooks ───────────────────────────────────────────────────────────

    def on_miss(self, obj_id: int) -> None:
        node = _Node(obj_id)
        self._link_at_head(node)
        self.nodes[obj_id] = node

    def on_hit(self, obj_id: int) -> None:
        node = self.nodes.get(obj_id)
        if node:
            node.visited = True

    def evict(self) -> int:
        if not self.nodes:
            return 0

        if self.hand is None:
            self.hand = self.tail

        # Walk toward head; reset visited items until an unvisited one is found.
        # Because we reset visited=False as we go, the loop always terminates
        # (the starting node will eventually be reached with visited=False).
        while self.hand.visited:
            self.hand.visited = False
            self.hand = self.hand.prev or self.tail  # wrap to tail at head

        victim = self.hand
        self.hand = victim.prev  # may be None; resets on next call
        self._unlink(victim)
        del self.nodes[victim.obj_id]
        return victim.obj_id

    def on_remove(self, obj_id: int) -> None:
        node = self.nodes.pop(obj_id, None)
        if node is None:
            return
        if self.hand is node:
            self.hand = node.prev  # advance before unlinking
        self._unlink(node)


# ── libcachesim hook functions ──────────────────────────────────────────────


def init_hook(common_cache_params: CommonCacheParams):
    return SieveCache(common_cache_params.cache_size)


def hit_hook(data: SieveCache, req: Request) -> None:
    data.on_hit(req.obj_id)


def miss_hook(data: SieveCache, req: Request) -> None:
    data.on_miss(req.obj_id)


def eviction_hook(data: SieveCache, req: Request) -> int:
    return data.evict()


def remove_hook(data: SieveCache, obj_id: int) -> None:
    data.on_remove(obj_id)


def free_hook(data: SieveCache) -> None:
    data.nodes.clear()
    data.head = data.tail = data.hand = None


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
        cache_name="sieve",
    )

    trace = Path(__file__).parent.parent / "data" / "cloudPhysicsIO.vscsi"
    reader = TraceReader(trace=str(trace), trace_type=TraceType.VSCSI_TRACE)
    req_miss_ratio, byte_miss_ratio = cache.process_trace(reader)
    print(f"Request miss ratio: {req_miss_ratio:.4f}")
    print(f"Byte miss ratio:    {byte_miss_ratio:.4f}")
