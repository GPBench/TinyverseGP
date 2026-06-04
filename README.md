# This is a cloned version of TinyverseGP that has been freezed to enable reproducibility.

- It serves to compare Tree-based GP (TGP) and Cartesian GP (CGP) for evolving boolean functions called AND_n and XOR_n
- Provides two simplified GP models that have been used in literature to perform runtime analysis
  - [src/analysis/models/simple_tgp.py](https://github.com/GPBench/TinyverseGP/blob/runtime-analysis/src/analysis/models/simple_tgp.py): implementation of simple (1+1)-TGP with HVL prime mutation operator
  - [src/analysis/models/simple_cgp.py](https://github.com/GPBench/TinyverseGP/blob/runtime-analysis/src/analysis/models/simple_cgp.py)`: implementation of simple (1+1)-CGP with SAM mutation operator
- The problem class implementation of AND_n and XOR_n
  - [src/analysis/benchmarks/boolean.py](https://github.com/GPBench/TinyverseGP/blob/runtime-analysis/src/analysis/benchmarks/boolean.py)
  

###  The results obtained with this version of TinyverseGP have been presented in work submitted to PPSN 2026

-  The corresponding paper has been accepted for inclusion in the conference proceedings
-  An arXiv version of the paper can be obtained here: 
  
