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

