"""
TinyLLM: A minimalistic LLM-based program synthesizer for TinyverseGP.

This module drives a Large Language Model (either a local Hugging Face model or a
remote OpenAI model) to generate Python functions that solve a given problem. The
generated code is extracted from the model response, executed in an isolated
process (to guard against endless loops), evaluated against the problem and the
best candidate found across several iterations is returned.
"""

import random
import re
import textwrap
import contextlib
import os
import time
from .llm import LLMInterface
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, logging
from dataclasses import dataclass
from types import FunctionType
from typing import Callable, Optional, Any
from openai import OpenAI
from multiprocessing import Process, Queue

logging.set_verbosity_error()


@dataclass
class LLMHyperparameters:
    """
    Hyperparameter configuration for :class:`TinyLLM`.

    :ivar model_id: Identifier of the model (Hugging Face repo id or OpenAI model name).
    :ivar temperature: Sampling temperature controlling generation randomness.
    :ivar max_new_tokens: Maximum number of tokens generated per local-model call.
    :ivar input_length: Number of leading values in each dataset tuple treated as inputs.
    :ivar train_dataset_limit: Maximum number of examples embedded into the prompt.
    :ivar error_penalty: Fitness assigned when evaluation fails or times out.
    :ivar openai_api_key: OpenAI API key; if empty the configured local model is used.
    :ivar timeout: Per-evaluation timeout in seconds for the worker process.
    :ivar max_time: Overall time budget (seconds) for the generation loop.
    :ivar max_iterations: Maximum number of generate/evaluate iterations.
    :ivar minimizing_fitness: Whether lower fitness values are better.
    :ivar useGPU: Whether to load the local model onto the GPU (CUDA) instead of the CPU.
    """

    model_id: str
    temperature: float
    max_new_tokens: int
    input_length: int
    train_dataset_limit: int
    error_penalty: float
    openai_api_key: str  # Optional, if not provided, the defined local model will be used
    timeout: int
    max_time: int
    max_iterations: int
    minimizing_fitness: bool
    useGPU: bool


def evaluate_worker(queue: Queue, problem: Any, func: Optional[Callable], context: Any) -> None:
    """
    Evaluate a generated function inside a separate process.

    Running the evaluation in its own process makes it possible to enforce a hard
    timeout and thus to prevent potential endless loops in generated code. The
    resulting fitness (or ``None`` on failure) is communicated back through the queue.

    :param queue: Inter-process queue used to return the evaluation result.
    :param problem: Problem instance providing the ``evaluate`` method.
    :param func: The generated candidate function to evaluate.
    :param context: The GP/LLM model context passed on to ``problem.evaluate``.
    :return: ``None``. The result is placed on ``queue`` instead.
    """
    try:
        # Silence any stdout/stderr produced by executing the generated function
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            result = problem.evaluate(func, context)
        queue.put(result)
    except Exception as e:
        # Any failure (e.g. runtime error in generated code) yields no result
        queue.put(None)


class TinyLLM(LLMInterface):
    """
    Minimalistic LLM-based program synthesizer.

    Implements the :class:`~src.llm.llm.LLMInterface` by building a prompt from a
    problem, invoking an LLM, extracting the generated function, evaluating it and
    keeping track of the best candidate found.

    :ivar problem: Problem instance the synthesizer optimizes against.
    :ivar hyperparameters: Configuration controlling model selection and the loop.
    :ivar prompt: Optional fixed prompt; if ``None`` a prompt is built from the dataset.
    :ivar generated_function: Most recently generated candidate function.
    :ivar best_fitness: Best fitness observed so far.
    :ivar best_str_code: Source code of the best candidate found so far.
    :ivar best_callable_code: Callable of the best candidate found so far.
    :ivar tokenizer_store: Cached tokenizer of the local model (lazy-loaded).
    :ivar model_store: Cached local model instance (lazy-loaded).
    """

    def __init__(
        self,
        problem_: object,
        hyperparameters_: LLMHyperparameters,
        prompt_: Optional[str] = None,
    ) -> None:
        """
        Create a new TinyLLM synthesizer.

        :param problem_: Problem instance to solve.
        :param hyperparameters_: Configuration for model selection and the loop.
        :param prompt_: Optional fixed prompt; if ``None`` a prompt is built from the dataset.
        :return: ``None``.
        """
        self.problem = problem_
        self.hyperparameters = hyperparameters_
        self.prompt = prompt_
        self.generated_function = None
        self.best_fitness = None
        self.best_str_code = None
        self.best_callable_code = None
        self.tokenizer_store = None  # cached tokenizer to avoid reloading the local model
        self.model_store = None  # cached local model instance

    def evaluate(self) -> float:
        """
        Evaluate the most recently generated function under a hard timeout.

        The evaluation runs in a dedicated process; if it does not finish within
        the configured timeout, the process is terminated and the error penalty
        is returned instead.

        :return: The evaluated fitness, or the configured error penalty on
            timeout/failure.
        """
        queue = Queue()
        p = Process(target=evaluate_worker, args=(queue, self.problem, self.generated_function, self))
        p.start()
        p.join(timeout=self.hyperparameters.timeout)

        # A still-alive process means the timeout was hit -> terminate and penalize
        if p.is_alive():
            p.terminate()
            p.join()
            return self.hyperparameters.error_penalty
        else:
            result = queue.get()
            return result if result is not None else self.hyperparameters.error_penalty

    def predict(self, model: Callable, observation: list) -> list:
        """
        Apply a generated function to a single observation.

        :param model: The generated callable to invoke.
        :param observation: Observation whose leading ``input_length`` values are
            passed as positional arguments to ``model``.
        :return: A single-element list holding the model's output.
        """
        # Only the configured number of leading values are treated as inputs
        inputs = observation[:self.hyperparameters.input_length]
        return [model(*inputs)]

    def build_prompt(self) -> str:
        """
        Build the prompt sent to the LLM.

        If a fixed prompt was provided at construction time it is used verbatim;
        otherwise a prompt is synthesized from a shuffled, truncated view of the
        problem's dataset.

        :return: The prompt string.
        """
        if self.prompt is not None:
            return self.prompt.strip() + "\n\n"

        def format_dataset(dataset: list, input_length: int) -> str:
            """
            Render dataset tuples as ``inputs -> outputs`` lines for the prompt.

            :param dataset: Iterable of example tuples (inputs followed by outputs).
            :param input_length: Number of leading values treated as inputs.
            :return: Newline-joined example lines, truncated to the dataset limit.
            """
            random.shuffle(dataset)  # avoid biasing the model towards a fixed ordering
            lines = []
            i = 0
            for tup in dataset:
                inputs = ", ".join(map(str, tup[:input_length]))
                outputs = ", ".join(map(str, tup[input_length:]))
                lines.append(f"{inputs} -> {outputs}")
                i += 1
                # Cap the number of examples to keep the prompt within a reasonable size
                if i >= self.hyperparameters.train_dataset_limit:
                    break
            return "\n".join(lines)

        prompt = "Given the input-output pairs:\n\n"
        prompt += format_dataset(self.problem.dataset, self.hyperparameters.input_length)
        prompt += "\n\nWrite a simple function named \"calculate\" for solving above input-output pairs that generalizes also to unseen data. " \
            + "Use only simple Python code without any imports and return only the requested function. Mark the code output as Python code."

        return prompt

    def extract_result(self, text: str) -> tuple[Optional[str], Optional[Callable]]:
        """
        Extract an executable function from a raw LLM response.

        The method first looks for a fenced ```python``` block and, failing that,
        for a bare ``def`` definition. The extracted source is executed in an
        isolated namespace and the first defined function is returned.

        :param text: Raw text returned by the LLM.
        :return: A ``(source_code, callable)`` tuple, or ``(None, None)`` if no
            usable function could be extracted.
        """
        try:
            # Prefer a fenced python code block if the model marked one
            match = re.search(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1).strip()
            else:
                # Fallback: grab the first bare function definition and its indented body
                match_plain = re.search(
                    r"(def\s+\w+\s*\(.*?\):(?:\n[ \t]+.*)+)", text, re.DOTALL
                )
                if not match_plain:
                    return None, None
                code = match_plain.group(1)

            code = textwrap.dedent(code)

            # Execute the snippet in an isolated namespace to materialize the function object
            namespace = {}
            exec(code, namespace, namespace)

            functions = {name: obj for name, obj in namespace.items() if isinstance(obj, FunctionType)}
            if not functions:
                return None, None

            return code, next(iter(functions.values()))
        except Exception as e:
            # print(f"Error extracting result: {e}")
            return None, None

    def invoke_lokal_llm(self, prompt: str) -> str:
        """
        Invoke the configured local Hugging Face model.

        The tokenizer and model are loaded lazily on first use and cached for
        subsequent calls. They are placed on the GPU or CPU depending on
        ``useGPU``.

        :param prompt: The prompt to feed to the model.
        :return: The generated text.
        """
        # Reuse the cached tokenizer/model if they have already been loaded
        if self.tokenizer_store is not None and self.model_store is not None:
            tokenizer = self.tokenizer_store
            model = self.model_store
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.hyperparameters.model_id)
            # Select the compute device based on the useGPU hyperparameter
            if self.hyperparameters.useGPU:
                model = AutoModelForCausalLM.from_pretrained(self.hyperparameters.model_id).to("cuda")
            else:
                model = AutoModelForCausalLM.from_pretrained(self.hyperparameters.model_id).to("cpu")
            self.tokenizer_store = tokenizer
            self.model_store = model

        # Ensure a padding token exists (required by the generation pipeline)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.hyperparameters.max_new_tokens,
            temperature=self.hyperparameters.temperature,
            pad_token_id=tokenizer.pad_token_id
        )

        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        return llm.invoke(prompt)

    def invoke_openai_llm(self, prompt: str) -> str:
        """
        Invoke a remote OpenAI chat model.

        :param prompt: The prompt to send as the user message.
        :return: The content of the model's response message.
        """
        client = OpenAI(api_key=self.hyperparameters.openai_api_key)

        response = client.chat.completions.create(
            model=self.hyperparameters.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.hyperparameters.temperature
        )
        return response.choices[0].message.content

    def generate(self) -> tuple[Optional[str], Optional[Callable]]:
        """
        Run the generate/evaluate loop and return the best candidate found.

        On every iteration a response is obtained from the OpenAI model (when an
        API key is configured) or from the local model, the returned function is
        extracted and evaluated and the best candidate is tracked. The loop stops
        after ``max_iterations`` or once the ``max_time`` budget is exhausted.

        :return: A ``(source_code, callable)`` tuple for the best candidate found.
        """
        prompt = self.build_prompt()

        t0 = time.time()
        elapsed = 0
        for _ in range(self.hyperparameters.max_iterations):
            # Prefer OpenAI when an API key is configured, otherwise use the local model
            if self.hyperparameters.openai_api_key.strip() not in [None, '']:
                response = self.invoke_openai_llm(prompt)
            else:
                response = self.invoke_lokal_llm(prompt)

            str_code, callable_code = self.extract_result(response)
            self.generated_function = callable_code

            fitness = self.evaluate()
            # Update the incumbent whenever a strictly better fitness is observed
            if self.best_fitness is None or (self.hyperparameters.minimizing_fitness and fitness < self.best_fitness) or (not self.hyperparameters.minimizing_fitness and fitness > self.best_fitness):
                self.best_fitness = fitness
                self.best_str_code = str_code
                self.best_callable_code = callable_code

            print(f">>> Fitness: {self.best_fitness}")

            # Accumulate wall-clock time and stop once the time budget is exceeded
            t1 = time.time()
            delta = t1 - t0
            t0 = t1
            elapsed += delta
            if elapsed + delta >= self.hyperparameters.max_time:
                break

        return self.best_str_code, self.best_callable_code
