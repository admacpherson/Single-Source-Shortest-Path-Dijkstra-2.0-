# Breaking the Sorting Barrier for Single-Source Shortest Paths (Dijkstra 2.0)

Based on research by [Duan et. al](https://arxiv.org/pdf/2504.17033)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Implementation of "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths" by Duan et al. (2025) - the first deterministic algorithm to achieve O(m log^(2/3) n) time complexity for SSSP, breaking Dijkstra's O(m + n log n) bound on sparse graphs.
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
│   ├── test_performance.py   # Runtime comparison
│   └── test_graphs.py        # Graph generation utilities
├── benchmarks/
│   ├── run_benchmarks.py     # Performance benchmarking script
│   └── visualize_results.py  # Results visualization
└── examples/
    └── demo.py               # Usage examples
```
## References

This implementation is based on the groundbreaking paper by Ran Duan, Jiayi Mao, Xiao Mao, Xinkai Shu, and Longhui Yin from Tsinghua University, Stanford University, and Max Planck Institute for Informatics.

```bibtex
@inproceedings{duan2025breaking,
  title={Breaking the Sorting Barrier for Directed Single-Source Shortest Paths},
  author={Duan, Ran and Mao, Jiayi and Mao, Xiao and Shu, Xinkai and Yin, Longhui},
  booktitle={arXiv preprint arXiv:2504.17033},
  year={2025}
}
```