from typing import List, Tuple, Dict
import random


class Graph:
    def __init__(self, n: int, directed: bool = True):
        self.n = n
        self.directed = directed
        self.adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        self.m = 0  # Number of edges

    def add_edge(self, u: int, v: int, weight: float):
        if weight < 0:
            raise ValueError("Edge weights must be non-negative")
        if u < 0 or u >= self.n or v < 0 or v >= self.n:
            raise ValueError(f"Vertices must be in range [0, {self.n})")

        self.adj[u].append((v, weight))
        self.m += 1

        if not self.directed:
            self.adj[v].append((u, weight))
            self.m += 1

    def neighbors(self, u: int) -> List[Tuple[int, float]]:
        return self.adj[u]

    def to_constant_degree(self) -> 'Graph':
        # Count total edges needed
        total_edges = sum(len(neighbors) for neighbors in self.adj)
        new_n = self.n + total_edges

        g = Graph(new_n, directed=self.directed)

        vertex_map = {}  # Maps (original_vertex, edge_index) to new vertex
        next_new_vertex = self.n

        for u in range(self.n):
            neighbors = self.adj[u]

            if len(neighbors) <= 2:
                # Keep as is
                for v, w in neighbors:
                    g.add_edge(u, v, w)
            else:
                # Create cycle for this vertex
                cycle_vertices = []
                for i, (v, w) in enumerate(neighbors):
                    new_v = next_new_vertex
                    next_new_vertex += 1
                    cycle_vertices.append(new_v)
                    vertex_map[(u, i)] = new_v

                # Connect cycle with zero weights
                for i in range(len(cycle_vertices)):
                    next_i = (i + 1) % len(cycle_vertices)
                    g.add_edge(cycle_vertices[i], cycle_vertices[next_i], 0.0)

                # Add actual edges
                for i, (v, w) in enumerate(neighbors):
                    g.add_edge(cycle_vertices[i], v, w)

        return g

    @staticmethod
    def random_graph(n: int, m: int, max_weight: float = 100.0,
                     seed: int = None) -> 'Graph':
        if seed is not None:
            random.seed(seed)

        g = Graph(n, directed=True)
        edges_added = 0
        attempts = 0
        max_attempts = m * 10

        while edges_added < m and attempts < max_attempts:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)

            if u != v:
                weight = random.uniform(0.1, max_weight)
                g.add_edge(u, v, weight)
                edges_added += 1

            attempts += 1

        return g

    @staticmethod
    def random_sparse_graph(n: int, avg_degree: float = 4.0,
                            max_weight: float = 100.0, seed: int = None) -> 'Graph':
        m = int(n * avg_degree)
        return Graph.random_graph(n, m, max_weight, seed)

    @staticmethod
    def grid_graph(rows: int, cols: int, directed: bool = True) -> 'Graph':
        n = rows * cols
        g = Graph(n, directed=directed)

        def idx(r: int, c: int) -> int:
            return r * cols + c

        for r in range(rows):
            for c in range(cols):
                u = idx(r, c)

                # Right edge
                if c + 1 < cols:
                    v = idx(r, c + 1)
                    g.add_edge(u, v, random.uniform(1.0, 10.0))

                # Down edge
                if r + 1 < rows:
                    v = idx(r + 1, c)
                    g.add_edge(u, v, random.uniform(1.0, 10.0))

        return g

    def __str__(self) -> str:
        return f"Graph(n={self.n}, m={self.m}, directed={self.directed})"

    def __repr__(self) -> str:
        return self.__str__()