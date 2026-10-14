# Parameters

## GPConfig

| **Name**              | **Type**                 | **Description**                                                                                  |
| --------------------- | ------------------------ | ------------------------------------------------------------------------------------------------ |
| `global_seed`         | int                      | Global seed for all jobs                                                                         |
| `num_jobs`            | int                      | Number of jobs                                                                                   |
| `max_generations`     | int                      | Maximum number of generations for the run                                                        |
| `stopping_criteria`   | float                    | -                                                                                                |
| `minimizing_fitness`  | bool                     | If `True`, smaller fitness values are better                                                     |
| `ideal_fitness`       | float                    | Ideal fitness value; run is stopped if fitness value better than or equal to this value is found |
| `silent_algorithm`    | bool                     | Switch for activating/deactivating generation reports                                            |
| `silent_evolver`      | bool                     | Switch for activating/deactivating job reports                                                   |
| `minimalistic_output` | bool                     | If `True`, job report only displays number of evaluations and best fitness of run                |
| `num_outputs`         | int                      | Number of program outputs                                                                        |
| `report_interval`     | int                      | Interval for generation report                                                                   |
| `max_time`            | int                      | Maximum time for the run in seconds                                                              |
| `constraints`         | Callable\[\[Any], float] | Constraint function that returns a `float` representing the degree of constraint violation       |
| `checkpointing`       | bool                     | Switch for activating/deactivating checkpointing                                                 |
| `checkpoint_interval` | int                      | Checkpointing interval based on the generation number                                            |
| `checkpoint_dir`      | str                      | Name of the checkpoint directory                                                                 |
| `experiment_name`     | str                      | Name for the experiment directory                                                                |

## TGP
#### TGPConfig

| **Name**    | **Type** | **Description**                                                                         |
| ----------- | -------- | --------------------------------------------------------------------------------------- |
| `max_size`  | int      | Maximum number of nodes per tree                                                        |
| `max_depth` | int      | Maximum depth of the trees                                                              |
| `erc`       | bool     | Switch to enable/disable use of ephemeral random constants in population initialisation |

## CGP
#### CGPConfig

| **Name**                   | **Type**       | **Description**                                |
| -------------------------- | -------------- | ---------------------------------------------- |
| `num_inputs`               | int            | Number of inputs                               |
| `num_outputs`              | int            | Number of outputs                              |
| `num_functions`            | int            | Number of functions                            |
| `max_arity`                | int            | Maximum function arity                         |
| `max_time`                   | int            | Maximum time for the run in seconds            |
| `report_every_improvement` | bool           | -                                              |

#### CGPHyperparameters

| Name                       | Type           | Description                                    |
| -------------------------- | -------------- | ---------------------------------------------- |
| `mu`                       | int            | Number of parent individuals                   |
| `lmbda`                    | int            | Number of offspring individuals                |
| `num_function_nodes`       | int            | Number of function nodes                       |
| `strict_selection`         | bool           | Switch to activate/deactivate strict selection |
| `levels_back`              | int            | Levels back                                    |
| `mutation_rate`            | float          | Mutation rate for the mutation operator        |
| `population_size`          | Optional\[int] | Size of the population                         |
| `mutation_rate_genes`      | int            | -                                              |

## LGP
#### LGPConfig

| **Name**                   | **Type** | **Description**                     |
| -------------------------- | -------- | ----------------------------------- |
| `num_registers`            | int      | Number of calculation registers     |
| `max_time`                 | int      | Maximum time for the run in seconds |
| `report_every_improvement` | bool     | -                                   |

#### LGPHyperparameters

| Name                   | Type  | Description                                                                    |
| ---------------------- | ----- | ------------------------------------------------------------------------------ |
| `mu`                   | int   | Number of parent individuals                                                   |
| `tournament_size`      | int   | Tournament size                                                                |
| `min_len`              | int   | Minimum length of genome                                                       |
| `max_len`              | int   | Maximum length of genome                                                       |
| `initial_max_len`      | int   | Maximum length of genome for individuals of the first generation               |
| `p_register`           | float | Probability of appending a calculation register to `operands`                  |
| `register_slack`       | int   | Number of additional registers added to `num_registers`                        |
| `macro_variation_rate` | float | Macro variation rate                                                           |
| `micro_variation_rate` | float | Micro variation rate                                                           |
| `insertion_rate`       | float | Insertion rate                                                                 |
| `max_segment`          | int   | Maximum segment length                                                         |
| `reproduction_rate`    | float | Reproduction rate                                                              |
| `branch_probability`   | float | Probability of using a branching operator                                      |
| `erc`                  | bool  | Switch to enable/disable use of ephemeral random constants for genome creation |
| `default_value`        | float | Default register value                                                         |
| `protection`           | float | Value which replaces invalid result in `predict`                               |

## GE

#### GEHyperparameters

| **Name**        | **Type** | **Description**                                                               |
| --------------- | -------- | ----------------------------------------------------------------------------- |
| `genome_length` | int      | Number of codons in a genome                                                  |
| `codon_size`    | int      | Exclusive upper bound for the integer value of a codon                        |
| `penalty_value` | float    | Fitness value assigned to genomes that cannot be mapped to a valid expression |
