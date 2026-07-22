"""
Example module to test the LLM-based synthesizer with program synthesis problems.

Attempts to synthesize a Python function solving the "power of two" problem
provided on Leetcode.com:

https://leetcode.com/problems/power-of-two/description/

The script configures the LLM hyperparameters, builds a program-synthesis problem
from a benchmark dataset, and runs :meth:`TinyLLM.generate` to obtain a candidate
solution which is then printed.
"""

from src.gp.problem import *
from src.benchmark.program_synthesis.ps_benchmark import PSBenchmark
from src.benchmark.program_synthesis.leetcode.power_of_two import *
from src.llm.tiny_llm import *


# LLM hyperparameters: model selection, sampling settings and the generation budget
hyperparameters = LLMHyperparameters(
    model_id = "Qwen/Qwen2.5-Coder-1.5B", #"gpt-4o", #"deepseek-ai/deepseek-coder-6.7b-instruct",
    temperature = 0.8,
    max_new_tokens = 1000,
    input_length = 1,
    train_dataset_limit = 15,    # Limit the number of training examples to avoid long prompts
    minimizing_fitness = False,
    error_penalty = 0,
    openai_api_key = "",    # Optional, if not provided, the defined local model will be used
    timeout = 25,
    max_time = 180,
    max_iterations = 15,
    useGPU = False
)

# Benchmark generator and the range of inputs used to build the training dataset
generator = gen_power_of_two
n = 10
m = 100

# Build the labelled dataset and wrap it in a program-synthesis problem
benchmark = PSBenchmark(generator, [n,m])
problem = ProgramSynthesis(benchmark.dataset)

# Optional fixed prompt describing the target function in natural language
prompt = "Write a Python function named \"calculate\" that takes a single input x and returns 1 if x is a power of 2, and 0 otherwise. Ensure the output is formatted as Python code."

# Instantiate the synthesizer and print the source code of the best candidate found
tinyllm = TinyLLM(problem, hyperparameters)    # Prompt is optional, if not provided, it will be built from the dataset
print(tinyllm.generate()[0])
