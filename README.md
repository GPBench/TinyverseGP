# This is a cloned version of TinyverseGP that has been freezed to enable reproducibility.

###  The results obtained with this version of TinyverseGP are presented in a paper accepted at NeurIPS 2026

> D.-C. Dang, R. Kalkreuth, and A. Opris. Runtime analysis of cartesian genetic programming on
MAX: A proven exponential speedup. In Proceedings of Advances in Neural Information Processing Systems (NeurIPS 2026), pp. (To appear), 2026

- It serves to compare Tree-based GP (TGP) and Cartesian GP (CGP) on the MAX problem
- Provides two simplified GP models that have been used in literature to perform runtime analysis
  - [src/analysis/models/simple_tgp.py](https://github.com/GPBench/TinyverseGP/blob/neurips-2026/src/analysis/models/simple_tgp.py): implementation of simple (1+1)-TGP with HVL prime mutation operator
  - [src/analysis/models/simple_cgp.py](https://github.com/GPBench/TinyverseGP/blob/neurips-2026/src/analysis/models/simple_cgp.py): implementation of simple (1+1)-CGP with SAM and probabilistic mutation operators
- The problem class implementation of MAX
  - [src/analysis/benchmarks/max/max.py](https://github.com/GPBench/TinyverseGP/blob/neurips-2026/src/analysis/benchmarks/max/max.py)
- Run scripts used for our experiments
  - [examples/analysis/max](https://github.com/GPBench/TinyverseGP/tree/neurips-2026/examples/analysis/max)
- Datafiles of the experiments, plot scripts and statistical evaluation 
  - [experiments/](https://github.com/GPBench/TinyverseGP/tree/neurips-2026/results)
  - [experiments/plot/](https://github.com/GPBench/TinyverseGP/tree/neurips-2026/experiments/plot)
  - [experiments/evaluation/](https://github.com/GPBench/TinyverseGP/tree/neurips-2026/experiments/plot)
 

