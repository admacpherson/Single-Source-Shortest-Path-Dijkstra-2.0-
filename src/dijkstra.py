"""Classical Dijkstra's algorithm with Fibonacci heap."""

import heapq
from typing import List, Tuple, Optional
from .graph import Graph


def dijkstra(g: Graph, source: int) -> Tuple[List[float], List[Optional[int]]]:
    """Dijkstra single-source shortest path algorithm.

    Uses Python's heapq which implements a binary heap (not Fibonacci heap).
    Time complexity: O(m log n) with binary heap, O(m + n log n) with Fibonacci heap.

    Args:
        g: Input graph
        source: Source vertex

    Returns:
        Tuple of (distances, predecessors)
        - distances[v]: shortest distance from source to v (inf if unreachable)
        - predecessors[v]: predecessor of v in shortest path tree (None if no path)
    """
    n = g.n
    distances = [float('inf')] * n
    predecessors = [None] * n
    distances[source] = 0.0

    # Priority queue: (distance, vertex)
    pq = [(0.0, source)]
    visited = [False] * n

    while pq:
        dist_u, u = heapq.heappop(pq)

        # Skip if already processed (may have outdated entry in heap)
        if visited[u]:
            continue

        visited[u] = True

        # Relax edges from u
        for v, weight in g.neighbors(u):
            new_dist = dist_u + weight

            if new_dist < distances[v]:
                distances[v] = new_dist
                predecessors[v] = u
                heapq.heappush(pq, (new_dist, v))

    return distances, predecessors


