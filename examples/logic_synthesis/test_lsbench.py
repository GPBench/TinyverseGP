from src.benchmark.logic_synthesis.lsbench.lsbench import LSBench, LSRegressor
from src.gp.tiny_cgp import CGPConfig, CGPHyperparameters
from src.gp.tiny_ge import GEHyperparameters
from src.gp.tiny_lgp import LGPConfig, LGPHyperparameters
from src.gp.tiny_tgp import TGPConfig, TGPHyperparameters
from src.gp.tinyverse import Var, GPConfig

MAXTIME = 3600  # 1 hour
MAXGEN = 100
POPSIZE = 100

lsbench = LSBench(data_dir_='../../data/logic_synthesis')
benchmarks = [lsbench.add4(),
              lsbench.mul3(),
              lsbench.alu4(),
              lsbench.count4(),
              lsbench.dec4(),
              lsbench.enc8(),
              lsbench.epar8(),
              lsbench.mcomp4(),
              lsbench.icomp5()
              ]

functions_reduced = ["AND", "OR", "BUFA", "NOT"]
functions_extended = ["AND", "OR", "BUFA", "NOT", "XOR", "NAND", "NOR", "XNOR"]
terminals = None

tgp_hyperparams = TGPHyperparameters(
    max_depth=8,
    max_size=100,
    pop_size=POPSIZE,
    tournament_size=3,
    mutation_rate=0.2,
    cx_rate=0.9,
    erc=None
)

cgp_hyperparams = CGPHyperparameters(
    mu=1,
    lmbda=4,
    num_function_nodes=10,
    strict_selection=True,
    mutation_rate=0.2,
    population_size=POPSIZE,
    levels_back=10,
)

lgp_hyperparams = LGPHyperparameters(
    mu=POPSIZE,
    macro_variation_rate=0.75,
    micro_variation_rate=0.25,
    insertion_rate=0.5,
    max_segment=15,
    reproduction_rate=0.5,
    branch_probability=0.0,
    p_register=0.5,
    max_len=200,
    initial_max_len=35,
    erc=False,
    default_value=0.0,
    protection=1e10,
    penalization_validity_factor=0.0
)

ge_hyperparams = GEHyperparameters(
    pop_size=POPSIZE,
    genome_length=100,
    codon_size=1000,
    cx_rate=0.9,
    mutation_rate=0.1,
    tournament_size=2,
    penalty_value=99999,
)

#   Set up configurations for TGP and CGP
tgp_config = TGPConfig(
    num_jobs=1,
    max_generations=MAXGEN,
    stopping_criteria=0,
    minimizing_fitness=True,
    ideal_fitness=0,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1000000,
    max_time=MAXTIME,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir="examples/checkpoint",
    experiment_name="lsbench_tgp",
)

cgp_config = CGPConfig(
    num_jobs=1,
    max_generations=MAXGEN,
    stopping_criteria=0,
    minimizing_fitness=True,
    ideal_fitness=0,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_functions=len(functions_reduced),
    max_arity=2,
    num_inputs=1,
    num_outputs=1,
    report_interval=10,
    max_time=MAXTIME,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir="examples/checkpoint",
    experiment_name="lsbench_cgp",
)

lgp_config = LGPConfig(
    num_jobs=1,
    max_generations=MAXGEN,
    stopping_criteria=0,
    minimizing_fitness=True,
    ideal_fitness=0,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    report_interval=100000000000,
    max_time=500,
    num_registers=8,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir="examples/checkpoint",
    experiment_name="srbench_lgp",
)

ge_config = GPConfig(
    num_jobs=1,
    max_generations=MAXGEN,
    stopping_criteria=0,
    minimizing_fitness=True,
    ideal_fitness=0,
    silent_algorithm=True,
    silent_evolver=True,
    minimalistic_output=True,
    num_outputs=1,
    report_interval=1,
    max_time=60,
    global_seed=42,
    checkpoint_interval=10,
    checkpoint_dir='examples/checkpoint',
    experiment_name='sr_ge'
)

# print("LSBench has been created!")

for bm in benchmarks:
    print(f"Running benchmark: {bm.name}")

    num_inputs = bm.benchmark.num_inputs
    num_outputs = bm.benchmark.num_outputs

    tgp_config.num_inputs = num_inputs
    tgp_config.num_outputs = num_outputs

    cgp_config.num_inputs = num_inputs
    cgp_config.num_outputs = num_outputs

    lgp_config.num_inputs = num_inputs
    lgp_config.num_registers = num_outputs
    lgp_config.num_outputs = num_outputs

    ge_config.num_outputs = num_outputs

    tt = bm.get_truth_table()

    functions = lsbench.get_fs(bm.name)
    terminals = [Var(index=i, name_="x" + str(+ i)) for i in range(num_inputs)]

    tgp = LSRegressor(
        representation_="TGP",
        config_=tgp_config,
        hyperparameters_=tgp_hyperparams,
        functions_=functions,
        terminals_=terminals
    )

    cgp = LSRegressor(
        "CGP",
        cgp_config,
        cgp_hyperparams,
        functions_=functions,
        terminals_=terminals
    )

    lgp = LSRegressor(
        "LGP",
        lgp_config,
        lgp_hyperparams,
        functions_=functions,
        terminals_=terminals
    )

    ge = LSRegressor(
        "GE",
        ge_config,
        ge_hyperparams,
        functions_=functions,
        terminals_=terminals
    )

    tgp.fit(X=tt.inputs, y=tt.outputs)
    print(f"tgp score: {tgp.score(tt.inputs, tt.outputs)}")
    cgp.fit(X=tt.inputs, y=tt.outputs)
    print(f"cgp score: {cgp.score(tt.inputs, tt.outputs)}")
    lgp.fit(X=tt.inputs, y=tt.outputs)
    print(f"lgp score: {lgp.score(tt.inputs, tt.outputs)}")
    ge.fit(X=tt.inputs, y=tt.outputs)
    if ge.is_valid():
        print(f"ge score: {ge.score(tt.inputs, tt.outputs)}")
    else:
        print(f"GE score cannot be calculated due to invalid genome")
