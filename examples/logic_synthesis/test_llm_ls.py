"""
Example module to test TGP with logic synthesis problems.

Evolves a symbolic expression for a boolean function that
is presented as a compressed or uncompressed truth table.

The logic synthesis benchmark files are located in the data folder.

The provided benchmarks are taken from the General Boolean Function Benchmark Suite (GFBS):

https://dl.acm.org/doi/10.1145/3594805.3607131
"""

from src.llm.tiny_llm import TinyLLM, LLMHyperparameters
from src.benchmark.logic_synthesis.ls_benchmark import LSBenchmark
from src.gp.problem import BlackBox
from src.gp.functions import AND, OR, NAND, NOR
from src.gp.loss import hamming_distance_bitwise
from src.gp.tinyverse import Var


hyperparameters = LLMHyperparameters(
    model_id = "Qwen/Qwen2.5-Coder-1.5B", #"gpt-4o", #"deepseek-ai/deepseek-coder-6.7b-instruct",
    temperature = 0.8,
    max_new_tokens = 1000,
    input_length = 10,
    train_dataset_limit = 15,    # Limit the number of training examples to avoid long prompts
    minimizing_fitness = False,
    error_penalty = 0,
    openai_api_key = "",    # Optional, if not provided, the defined local model will be used
    timeout = 25,
    max_time = 180,
    max_iterations = 15,
    useGPU = False
)

benchmark = LSBenchmark('data/logic_synthesis/plu/add3.plu')
benchmark.generate()
truth_table = benchmark.get_truth_table()
num_inputs = benchmark.benchmark.num_inputs
num_outputs = benchmark.benchmark.num_outputs

functions = [AND, OR, NAND, NOR]
terminals = [Var(i) for i in range(num_inputs)]

prompt = "Given the following input-output table:\n\n"
for xs, ys in zip(truth_table.inputs, truth_table.outputs):
    x = ', '.join(xs)
    y = ', '.join(ys)
    prompt += f"{x} -> {y}\n"
prompt += "\n\nWrite a simple function named \"calculate\" for solving above input-output pairs that generalizes also to unseen data. " \
            + "Use only  the operators &,|,~ and return only the requested function. Mark the code output as Python code."

data = truth_table.inputs
actual = truth_table.outputs
loss = hamming_distance_bitwise
problem = BlackBox(data, actual, loss, 0, True)


tinyllm = TinyLLM(problem, hyperparameters, prompt)    # Prompt is optional, if not provided, it will be built from the dataset
print(tinyllm.generate()[0])
