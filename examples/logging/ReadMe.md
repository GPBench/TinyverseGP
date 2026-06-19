# Logging TinyverseGP Runs With IOH

This directory shows how to log TinyverseGP runs with
[IOHexperimenter (IOH)](https://iohprofiler.github.io/IOHexperimenter/).

Relevant examples in this folder:

- `examples/logging/test_tgp_sr_logged.py`
- `examples/logging/test_srbench_logged.py`
- `examples/logging/test_ge_ps_logged.py`

## 1. Create an IOH logger

Use `ioh.logger.Analyzer` and choose where to store logs.

```python
import ioh

logger = ioh.logger.Analyzer(
	[ioh.logger.trigger.ALWAYS],
	root=".",
	folder_name="test_logging",
	algorithm_name="TGP",
)
```

The configuration above logs every evaluation and writes IOH output files
into `./test_logging/`.

## 2. Attach the logger to a TinyverseGP problem

For problem classes that accept a `logger` argument (for example `BlackBox`),
pass the logger when creating the problem, together with a function number and name:

```python
from src.gp.problem import BlackBox

problem = BlackBox(
	observations_=data,
	actual_=actual,
	loss_=loss,
	ideal_=1e-6,
	minimizing_=True,
	fid=1,
	name="KOZA1",
	logger=logger,
)
```

Then run your GP model as usual:

```python
tgp = TinyTGP(functions, terminals, config, hyperparameters)
best = tgp.evolve(problem)
```

If you want to run multiple repetitions, make sure to call problem.reset() between them to ensure proper tracking. 

If running an experiment on more than one problem, such as in the SRBench example (`test_srbench_logged.py`), the logger can be passed to
`model.fit(...)`, which internally creates/uses the problem object with IOH logging enabled.

## 3. Process the generated data

You can analyze the generated IOH logs with either:

- [IOHanalyzer web app](https://iohprofiler.github.io/IOHanalyzer/)
- [`iohinspector` Python package](https://iohinspector.readthedocs.io/)

### Example with `iohinspector`

```python
import iohinspector as ii

# Load a folder produced by ioh.logger.Analyzer
df = ii.DataManager("./test_logging")

# Inspect loaded runs/metadata
print(df.overview)
```

If you prefer a GUI workflow, upload the same folder (zipped) to IOHanalyzer and use its visualization methods. 