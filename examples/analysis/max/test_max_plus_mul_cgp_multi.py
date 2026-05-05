"""
Run script to apply CGP to the MAX problem.

This is a multi-instance run script that performs n instances for each setting of D in the
closed interval [d_min, d_max].

"""

import numpy as np
import seaborn as sns
import pandas as pd
import csv
from matplotlib import pyplot as plt
from src.analysis.models.simple_cgp import SimpleCGP, MutationType, SimpleCGPConfig
from src.analysis.benchmarks.max.max import MaxPlusMul
from src.gp.tiny_cgp import *
from src.gp.functions import ADD, MUL
from src.gp.tinyverse import Const

NUM_INSTANCES = 30
MAX_GENERATIONS = 2000000
MAX_TIME = 9999999
EXPORT_CSV = False
PLOT = False
D_MIN = 1
D_MAX = 8
T = 1
functions = [ADD, MUL]
terminals = [Const(T)]

sns.set_theme()

config = SimpleCGPConfig(
    num_jobs=1,
    max_generations=MAX_GENERATIONS,
    stopping_criteria=None,
    minimizing_fitness=False,
    ideal_fitness=None,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions),
    max_arity=2,
    num_inputs=1,
    num_outputs=1,
    report_interval=1,
    max_time=MAX_TIME,
    mutation_type=MutationType.SAM,
    global_seed=None,
    checkpoint_interval=9999999,
    checkpoint_dir='../checkpoint',
    experiment_name='max_tgp'
)

hyperparameters = CGPHyperparameters(
    mu=1,
    lmbda=1,
    population_size=2,
    num_function_nodes=D_MAX,
    levels_back=D_MAX,
    mutation_rate=None,
    strict_selection=True,
)

x = []
y = []
csv_data = []
for d in range(D_MIN,D_MAX+1):
    iters = []
    deltas = []
    problem = MaxPlusMul(d=d, t=T)
    for _ in range(NUM_INSTANCES):
        config.ideal_fitness = problem.ideal
        config.global_seed = int(time.time_ns())
        cgp = SimpleCGP(functions, terminals, config, hyperparameters)
        t0 = time.time()
        best = cgp.evolve(problem)
        t1 = time.time()
        delta = t1 - t0
        iters.append(cgp.generation_number)
        deltas.append(delta)
        csv_data.append({'d': d, 'num_iters': cgp.num_evaluations})
        #print(f"{d},simple_tgp,{cgp.num_evaluations}")

    avg_iters = np.mean(iters)
    std = np.std(iters)
    avg_delta = np.mean(deltas)
    x.append(d)
    y.append(avg_iters)
    print(f"{d};{avg_iters:.2f};{std:.2f};{avg_delta:.2f}")

if EXPORT_CSV:
    with open('../max_plus_mul_cgp.csv', 'w', newline='') as csvfile:
        fieldnames = ['d', 'num_iters']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

if PLOT:
    data = pd.read_csv('../max_plus_mul_cgp.csv')

    p = sns.lineplot(
        data=data,
        x="d", y="num_iters",
        markers=True,
    )

    p.set(xlabel='D', ylabel='# Iterations')
    p.set_xticks(range(D_MIN,D_MAX+1))
    p.set_xticklabels([str(d) for d in range(D_MIN,D_MAX+1)])
    plt.show()
