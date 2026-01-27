# Breaking the Sorting Barrier for Single-Source Shortest Paths (Dijkstra 2.0)

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=fff)](#)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implementation of "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths" by Duan et al. (2025) - the first deterministic algorithm to achieve O(m log^(2/3) n) time complexity for SSSP, breaking Dijkstra's O(m + n log n) bound on sparse graphs. This algorithm is the first to prove that Dijkstra's algorithm is not optimal for single-source shortest paths on directed graphs with real non-negative edge weights.
The method combines two classical approaches:
- **Dijkstra's Algorithm**: Uses a priority queue to extract minimum distance vertices (sorting bottleneck: Ω(n log n))
- **Bellman-Ford Algorithm**: Relaxes all edges iteratively without sorting (O(mk) for paths with ≤k edges)

The new algorithm uses recursive partitioning to reduce frontier size by a factor of log^Ω(1)(n), avoiding the full sorting bottleneck.

### Time Complexity

| Algorithm                 | Time Complexity                                        | Graph Type                     |
|---------------------------|--------------------------------------------------------|--------------------------------|
| Dijkstra (Fibonacci Heap) | O(m + n log n)                                         | Directed, non-negative weights |
| **Duan et. al**           | **O(m log^(2/3) n)**                                   | Directed, non-negative weights |
| Improvement               | **Faster on sparse graphs** where m = o(n log^(1/3) n) | -                              |

## Repository Structure

```
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── .gitignore                 # gitignore
├── src/
│   ├── __init__.py
│   ├── dijkstra.py           # Classical Dijkstra implementation
│   ├── duan_algorithm.py     # Implementation of new algorithm
│   ├── graph.py              # Graph data structure
│   └── utils.py              # Helper functions
├── tests/
│   ├── __init__.py
│   ├── test_correctness.py   # Correctness verification
│   ├── test_graphs.py        # Graph generation utilities
│   ├── test_performance.py   # Runtime comparison
│   └── test_utils.py         # Utility function tests

├── benchmarks/
│   ├── run_benchmarks.py     # Performance benchmarking script
│   └── visualize_results.py  # Results visualization
└── examples/
    ├── demo.py               # Basic usage examples
    └── advanced_demo.py      # Advanced features demonstration
```
## References

This implementation is based on the groundbreaking paper by Ran Duan, Jiayi Mao, Xiao Mao, Xinkai Shu, and Longhui Yin from Tsinghua University, Stanford University, and Max Planck Institute for Informatics.

**Paper:** [arXiv:2504.17033](https://arxiv.org/abs/2504.17033)

```bibtex
@inproceedings{duan2025breaking,
  title={Breaking the Sorting Barrier for Directed Single-Source Shortest Paths},
  author={Duan, Ran and Mao, Jiayi and Mao, Xiao and Shu, Xinkai and Yin, Longhui},
  booktitle={arXiv preprint arXiv:2504.17033},
  year={2025}
}
```