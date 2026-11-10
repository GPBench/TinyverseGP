# Profiling notes (initial)

- Target example: `examples/symbolic_regression.test_tgp_sr` (or closest available)
- Env: Ubuntu on WSL2, Python: `python --version`

Next:
- Run: `python -m cProfile -o prof.out -m examples.symbolic_regression.test_tgp_sr`
- Inspect: `python -c "import pstats; p=pstats.Stats('prof.out'); p.sort_stats('cumtime').print_stats(20)"`
