# Using the SMAC3 interface for HPO

When performing an experiment, you might want to optimise the hyperparameters. This tutorial demonstrates the use of the [SMAC3](https://github.com/automl/SMAC3/) interface for HPO on a symbolic regression problem using CGP. For more information on how to perform experiments using a different representation or problem domain, see [Running a Symbolic Regression (SR) experiment](sr_tutorial.md).

## 1. Imports

```python
# CGP imports
from gp.tiny_cgp import *
from gp.problem import BlackBox
from benchmark.symbolic_regression.sr_benchmark import SRBenchmark
from benchmark.symbolic_regression.srbench import SRBench
from gp.functions import *
from gp.loss import *
from gp.tinyverse import Var, Const

# Additional imports for the dataset and HPO
import numpy as np
from sklearn.model_selection import train_test_split
from pmlb import fetch_data
from hpo.hpo import SMACInterface
import sys
import copy
```

## 2. Preparation

For reproducibility of the HPO, set a seed: 

```python
# Seed for HPO
smac_seed = 42
```

The evolutionary run can be made reproducible by setting a seed as well. In this tutorial, we will be using a dataset from the [Penn Machine Learning Benchmarks (PMLB)](https://epistasislab.github.io/pmlb/) dataset collection .

```python
seed = 67
dataset = "192_vineyard"
```

Get the data:

```python
X, y = fetch_data(dataset, return_X_y=True)

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75, random_state=smac_seed)
```

Next, set the function and terminal set, GP configuration, and initial hyperparameters you want to use.

```python

# Set function set and terminal set
functions = ['+','-','*','/','exp','log','square','cube']
terminals=[1,0.5,np.pi, np.sqrt(2)]

X, y = fetch_data(dataset, return_X_y=True)

# Configuration for the CGP evolutionary run
cgp_config = CGPConfig(
    num_jobs=1,
    max_generations=200,
    stopping_criteria=1e-6,
    minimizing_fitness=True,
    ideal_fitness=1e-6,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=X.shape[1],
    num_outputs=1,
    report_interval=1,
    max_time=600,
    global_seed=smac_seed
)

# Hyperparameter settings for CGP initialisation
cgp_hyperparams = CGPHyperparameters(
    mu=1,
    lmbda=99,
    population_size=1,
    levels_back=100,
    mutation_rate=0.1,
    strict_selection=True,
    cx_rate = 0.7,
    tournament_size = 7,
    num_function_nodes = 10,
    operator = operator_
)
```

## 3. HPO

Optimise the hyperparameters of our GP model for the selected dataset:

```python
# CGP initialisation
cgp = SRBench('CGP', cgp_config, cgp_hyperparams, functions=functions, terminals=terminals, scaling_=False)
cgp.fit(train_X, train_y)

# SMAC3 settings
n_trials = 200

# HPO
interface = SMACInterface()
opt_hyperparameters = interface.optimise(gpmodel_=cgp.model, dataset_=dataset, train_X_=train_X, train_y_=train_y, n_trials_=n_trials, seed_=seed, n_splits_=n_splits)
```

The optimised configuration is assigned to `opt_hyperparameters`.
