import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph import Graph
from src.dijkstra import dijkstra
from src.duan_algorithm import duan_sssp


def demo_simple_graph():
    print("=" * 80)
    print("DEMO: Simple Graph Example")
    print("=" * 80)

    # Create a simple graph
    g = Graph(6, directed=True)
    edges = [
        (0, 1, 7.0),
        (0, 2, 9.0),
        (0, 5, 14.0),
        (1, 2, 10.0),
        (1, 3, 15.0),
        (2, 3, 11.0),
        (2, 5, 2.0),
        (3, 4, 6.0),
        (4, 5, 9.0),
    ]

    print("\nGraph structure:")
    print("Vertices: 6 (labeled 0-5)")
    print("Edges:")
    for u, v, w in edges:
        g.add_edge(u, v, w)
        print(f"  {u} -> {v} (weight {w})")

    source = 0
    print(f"\nSource vertex: {source}")

    # Run Dijkstra
    print("\n" + "-" * 80)
    print("Running Dijkstra's Algorithm...")
    distances_dijk, pred_dijk = dijkstra(g, source)

    print("\nShortest distances from source:")
    for v in range(g.n):
        if distances_dijk[v] == float('inf'):
            print(f"  Vertex {v}: unreachable")
        else:
            print(f"  Vertex {v}: {distances_dijk[v]:.1f}")

    # Run Duan
    print("\n" + "-" * 80)
    print("Running Duan et al. Algorithm...")
    distances_duan, pred_duan = duan_sssp(g, source)

    print("\nShortest distances from source:")
    for v in range(g.n):
        if distances_duan[v] == float('inf'):
            print(f"  Vertex {v}: unreachable")
        else:
            print(f"  Vertex {v}: {distances_duan[v]:.1f}")

    # Verify they match
    print("\n" + "-" * 80)
    print("Verification:")
    all_match = True
    for v in range(g.n):
        if abs(distances_dijk[v] - distances_duan[v]) > 1e-9:
            print(f"  ❌ Vertex {v}: Mismatch!")
            all_match = False

    if all_match:
        print("  ✓ All distances match!")

    print()


