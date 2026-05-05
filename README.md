# This is a cloned version of TinyverseGP that has been anonymized and freezed for peer-review. 

- It serves to compare Tree-based GP (TGP) and Cartesian GP (CGP) on the MAX problem
- Provides two simplified GP models that have been used in literature to perform runtime analysis
  - [src/analysis/models/simple_tgp.py](https://github.com/GPBench/TinyverseGP/blob/max-analysis/src/analysis/models/simple_tgp.py): implementation of simple (1+1)-TGP with HVL prime mutation operator
  - [src/analysis/models/simple_cgp.py](https://github.com/GPBench/TinyverseGP/blob/max-analysis/src/analysis/models/simple_cgp.py)`: implementation of simple (1+1)-CGP with SAM and probabilistic mutation operators
- The problem class implementation of MAX
  - [src/analysis/benchmarks/max/max.py](https://github.com/GPBench/TinyverseGP/blob/max-analysis/src/analysis/benchmarks/max/max.py)
- Run scripts used for our experiments
  - [examples/analysis/max](https://github.com/GPBench/TinyverseGP/tree/max-analysis/examples/analysis/max)
- Datafiles of the experiments in CSV format and the plot script 
  - [results/](https://github.com/GPBench/TinyverseGP/tree/max-analysis/results)

