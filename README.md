# This is a cloned version of TinyverseGP that has been anonymized.

- It serves to compare Tree-based GP (TGP) and Cartesian GP (CGP) for evolving boolean functions called AND_n and XOR_n
- Provides two simplified GP models that have been used in literature to perform runtime analysis
  - `src/analysis/models/tiny_tgp.py`: implementation of simple (1+1)-TGP with HVL prime mutation
  - `src/analysis/models/tiny_cgp.py`: implementation of simple (1+1)-CGP with SAM mutation
- The problem class implementation of AND_n and XOR_n
  - `src/analysis/benchmarks/boolean.py`

  

