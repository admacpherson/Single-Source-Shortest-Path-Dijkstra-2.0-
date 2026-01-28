"""Utility functions for SSSP algorithms."""

from typing import List, Optional, Set, Dict
import math
from .graph import Graph


def reconstruct_path(predecessors: List[Optional[int]], source: int, target: int) -> List[int]:
    """Reconstruct shortest path from source to target using predecessor array.

    Args:
        predecessors: Predecessor array from SSSP algorithm
        source: Source vertex
        target: Target vertex

    Returns:
        List of vertices in path from source to target, or empty list if no path exists
    """
    if predecessors[target] is None and target != source:
        return []  # No path exists

    path = []
    current = target

    # Work backwards from target to source
    while current is not None:
        path.append(current)
        if current == source:
            break
        current = predecessors[current]

    # Reverse to get source -> target order
    path.reverse()

    # Verify path is valid
    if path[0] != source:
        return []  # Path doesn't start at source

    return path


def compute_path_length(g: Graph, path: List[int]) -> float:
    """Compute total length of a path.

    Args:
        g: Graph
        path: List of vertices forming a path

    Returns:
        Total weight of path, or inf if path is invalid
    """
    if len(path) <= 1:
        return 0.0

    total = 0.0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        # Find edge weight
        found = False
        for neighbor, weight in g.neighbors(u):
            if neighbor == v:
                total += weight
                found = True
                break

        if not found:
            return float('inf')  # Edge doesn't exist

    return total


def verify_path(g: Graph, path: List[int], expected_distance: float) -> bool:
    """Verify that a path has the expected distance.

    Args:
        g: Graph
        path: List of vertices
        expected_distance: Expected total distance

    Returns:
        True if path distance matches expected distance
    """
    actual_distance = compute_path_length(g, path)
    return abs(actual_distance - expected_distance) < 1e-9


def print_path(path: List[int], distances: List[float] = None):
    """Pretty print a path.

    Args:
        path: List of vertices in path
        distances: Optional distances array to show cumulative distances
    """
    if not path:
        print("No path exists")
        return

    print("Path: ", end="")
    for i, v in enumerate(path):
        print(v, end="")
        if distances is not None and i < len(path) - 1:
            print(f" [{distances[v]:.2f}] -> ", end="")
        elif i < len(path) - 1:
            print(" -> ", end="")

    if distances is not None:
        print(f" [{distances[path[-1]]:.2f}]")
    else:
        print()


def compute_reachability(g: Graph, source: int) -> Set[int]:
    """Compute set of vertices reachable from source using BFS.

    Args:
        g: Graph
        source: Source vertex

    Returns:
        Set of reachable vertices
    """
    reachable = {source}
    queue = [source]
    front = 0

    while front < len(queue):
        u = queue[front]
        front += 1

        for v, _ in g.neighbors(u):
            if v not in reachable:
                reachable.add(v)
                queue.append(v)

    return reachable

def compare_distances(distances1: List[float], distances2: List[float],
    if len(distances1) != len(distances2):
        return False

    for d1, d2 in zip(distances1, distances2):
        # Both infinite
        if d1 == float('inf') and d2 == float('inf'):
            continue

        # One infinite, one finite
        if d1 == float('inf') or d2 == float('inf'):
            return False

        # Both finite but different
        if abs(d1 - d2) > epsilon:
            return False

    return True

def theoretical_complexity(n: int, m: int, algorithm: str) -> float:
    if algorithm.lower() == 'dijkstra':
        # O(m + n log n) with Fibonacci heap
        return m + n * math.log2(n) if n > 0 else 0

    elif algorithm.lower() == 'duan':
        # O(m log^(2/3) n)
        if n <= 1:
            return m
        return m * math.pow(math.log2(n), 2 / 3)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def theoretical_speedup(n: int, m: int) -> float:
    dijkstra_ops = theoretical_complexity(n, m, 'dijkstra')
    duan_ops = theoretical_complexity(n, m, 'duan')

    if duan_ops == 0:
        return float('inf')

    return dijkstra_ops / duan_ops


def format_distance(distance: float, decimals: int = 2) -> str:
    """Format a distance for display.

    Args:
        distance: Distance value
        decimals: Number of decimal places

    Returns:
        Formatted string
    """
    if distance == float('inf'):
        return "∞"
    return f"{distance:.{decimals}f}"


def print_distance_table(distances: List[float], source: int,
                         vertices_per_row: int = 10):
    """Print distances in a formatted table.

    Args:
        distances: Distance array
        source: Source vertex
        vertices_per_row: Number of vertices to show per row
    """
    n = len(distances)

    print(f"Shortest distances from vertex {source}:")
    print("-" * 50)

    for start in range(0, n, vertices_per_row):
        end = min(start + vertices_per_row, n)

        # Print vertex numbers
        print("  v: ", end="")
        for v in range(start, end):
            print(f"{v:>8}", end="")
        print()

        # Print distances
        print("  d: ", end="")
        for v in range(start, end):
            print(f"{format_distance(distances[v], 2):>8}", end="")
        print()
        print()

