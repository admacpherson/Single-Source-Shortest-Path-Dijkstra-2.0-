"""
Implementation of the algorithm from:
"Breaking the Sorting Barrier for Directed Single-Source Shortest Paths"
by Duan, Mao, Mao, Shu, and Yin (2025)

Time complexity: O(m log^(2/3) n)
"""

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

