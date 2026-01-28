"""
Implementation of the algorithm from:
"Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"
by Duan, Mao, Mao, Shu, and Yin (2025)

Time complexity: O(m log^(2/3) n)
"""

import heapq
from typing import List, Tuple, Optional, Set, Dict
from .graph import Graph


class PartialSortDS:
    """Partial sorting data structure from Lemma 3.3."""

    def __init__(self, M: int, B: float):
        """Initialize data structure.

        Args:
            M: Maximum number of elements to pull at once
            B: Upper bound on all values
        """
        self.M = M
        self.B = B
        self.items: Dict[int, float] = {}  # key -> value mapping

    def insert(self, key: int, value: float):
        """Insert a key/value pair, keeping minimum value for each key."""
        if key not in self.items or value < self.items[key]:
            self.items[key] = value

    def batch_prepend(self, pairs: List[Tuple[int, float]]):
        """Batch prepend multiple key/value pairs."""
        for key, value in pairs:
            if key not in self.items or value < self.items[key]:
                self.items[key] = value

    def pull(self) -> Tuple[float, Set[int]]:
        """Return at most M keys with smallest values and separation bound.

        Returns:
            (B_i, S_i) where B_i separates S_i from remaining keys
        """
        if not self.items:
            return self.B, set()

        # Sort items by value
        sorted_items = sorted(self.items.items(), key=lambda x: x[1])

        if len(sorted_items) <= self.M:
            # Return all items
            keys = set(k for k, v in sorted_items)
            self.items.clear()
            return self.B, keys

        # Return first M items
        selected = sorted_items[:self.M]
        keys = set(k for k, v in selected)

        # Upper bound is the smallest remaining value
        upper_bound = sorted_items[self.M][1]

        # Remove selected items
        for key, _ in selected:
            del self.items[key]

        return upper_bound, keys

    def is_empty(self) -> bool:
        """Check if data structure is empty."""
        return len(self.items) == 0


def find_pivots(g: Graph, B: float, S: Set[int],
                db: List[float], k: int) -> Tuple[Set[int], Set[int]]:
    """FindPivots procedure from Algorithm 1

    Args:
        g: Input graph
        B: Upper bound
        S: Set of vertices
        db: Current distance estimates (modified in place)
        k: Parameter (floor(log^(1/3)(n)))

    Returns:
        (P, W) where P is set of pivots and W is set of vertices visited
    """
    W = set(S)
    W_i = set(S)  # W_0 in the paper

    # Relax for k steps (lines 4-11)
    for i in range(1, k + 1):
        W_next = set()

        # For all edges (u,v) with u in W_{i-1}
        for u in W_i:
            for v, w_uv in g.neighbors(u):
                new_dist = db[u] + w_uv

                # Line 7: if db[u] + w_uv <= db[v]
                if new_dist <= db[v]:
                    # Line 8: db[v] <- db[u] + w_uv
                    db[v] = new_dist

                    # Lines 9-10: if db[u] + w_uv < B then
                    if new_dist < B:
                        W_next.add(v)
                        W.add(v)

        # Line 12-14: Early termination if W becomes too large
        if len(W) > k * len(S):
            P = S
            return P, W

        W_i = W_next

    # Lines 15-16: Build forest F
    # F = {(u,v) in E : u,v in W, db[v] = db[u] + w_uv}
    # Under Assumption 2.1, F is a directed forest
    parent = {}
    for v in W:
        parent[v] = None

    for u in W:
        for v, w_uv in g.neighbors(u):
            if v in W and abs(db[v] - (db[u] + w_uv)) < 1e-9:
                parent[v] = u

    # Compute subtree sizes
    def get_subtree_size(v: int, visited: Set[int]) -> int:
        """Count vertices in subtree rooted at v."""
        if v in visited:
            return 0
        visited.add(v)

        size = 1
        for u in W:
            if parent.get(u) == v:
                size += get_subtree_size(u, visited)
        return size

    # Line 16: P = {u in S : u is root of tree with >= k vertices in F}
    P = set()
    for u in S:
        if u in W:
            visited = set()
            size = get_subtree_size(u, visited)
            if size >= k:
                P.add(u)

    return P, W


def base_case_bmssp(g: Graph, B: float, x: int,
                    db: List[float], k: int) -> Tuple[float, Set[int]]:
    """BaseCase procedure from Algorithm 2 (l=0 case of BMSSP).

    Args:
        g: Input graph
        B: Upper bound
        x: Source vertex (S is singleton {x})
        db: Current distance estimates (modified in place)
        k: Parameter

    Returns:
        (B', U) where B' is boundary and U is set of complete vertices
    """
    # Line 2: U_0 <- S = {x}
    U = {x}

    # Line 3: Initialize heap with single element
    H = [(db[x], x)]
    visited = set()

    # Lines 4-13: While loop
    while H and len(U) < k + 1:
        # Line 5: Extract minimum
        dist_u, u = heapq.heappop(H)

        if u in visited:
            continue
        visited.add(u)

        # Line 6: U_0 <- U_0 ∪ {u}
        U.add(u)

        # Line 7: For each edge e = (u,v)
        for v, w_uv in g.neighbors(u):
            new_dist = db[u] + w_uv

            # Line 8: if db[u] + w_uv <= db[v] and db[u] + w_uv < B
            if new_dist <= db[v] and new_dist < B:
                # Line 9: db[v] <- db[u] + w_uv
                db[v] = new_dist

                # Lines 10-13: Update heap
                if v not in visited:
                    heapq.heappush(H, (db[v], v))

    # Lines 14-17: Return based on size of U
    if len(U) <= k:
        # Line 15: return B' <- B, U <- U_0
        return B, U
    else:
        # Line 17: return B' <- max_{v in U_0} db[v], U <- {v in U_0 : db[v] < B'}
        B_prime = max(db[v] for v in U)
        U_filtered = {v for v in U if db[v] < B_prime}
        return B_prime, U_filtered

