"""
Example module to test TGP with symbolic regression problems.

Attempts to evolve a solution for the Koza-1 benchmkark which is
a quartic polynomial: x^4 + x^3 + x^2 + x

The problem is specified in the following paper:
https://dl.acm.org/doi/10.1145/2330163.2330273

Please note: This benchmark is nowadays considered a toy problem and
no serious benchmark. It only serves as an example for SR as an application
domain for TinyverseGP:
"""

from src.llm.tiny_llm import *
from src.gp.functions import *
from src.gp.loss import *
from src.gp.problem import BlackBox
from src.benchmark.symbolic_regression.sr_benchmark import SRBenchmark
from src.gp.tinyverse import Var, Const


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

loss = absolute_distance
benchmark = SRBenchmark()
data, actual = benchmark.generate("KOZA1")
functions = [ADD, SUB, MUL, DIV]
terminals = [Var(0), Const(1)]

problem = BlackBox(data, actual, loss, 1e-6, True)

prompt = "Given the input-output pairs:\n\n"
for x, y in zip(data[:20], actual[:20]):
    prompt += f"{x} -> {y}\n"
prompt += "\n\nWrite a simple function named \"calculate\" for solving above input-output pairs that generalizes also to unseen data. " \
            + "Use only  +, -, *, / and constants values and return only the requested function. Mark the code output as Python code."

tinyllm = TinyLLM(problem, hyperparameters, prompt)    # Prompt is optional, if not provided, it will be built from the dataset
print(tinyllm.generate()[0])
