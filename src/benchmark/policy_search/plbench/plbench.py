from sklearn.base import RegressorMixin
from src.benchmark.benchmark import Benchmark
from minatar import gym
from src.benchmark.policy_search.pl_benchmark import PLBenchmark
from src.gp import util
from src.gp.functions import ADD, SUB, MUL, DIV, AND, OR, NAND, NOR, NOTA, LT, GT, EQ, MIN, MAX, IF
from src.gp.problem import PolicySearch

strfun = {
    "+": ADD,
    "-": SUB,
    "*": MUL,
    "/": DIV,
    "AND": AND,
    "OR": OR,
    "NOTA": NOTA,
    "NAND": NAND,
    "NOR": NOR,
    "LT": LT,
    "GT": GT,
    "EQ": EQ,
    "MIN": MIN,
    "MAX": MAX,
    "IF": IF
}


class PLBench(Benchmark):

    def __init__(self):
        gym.register_envs()
        self.generate(args=None)

    def generate(self, args: any):
        self.benchmark = {"minatar": PLBench.MinAtar().problems}

    class MinAtar:
        def __init__(self, use_minimal_action_set_=True):
            self.problems = {
                "asterix": PLBenchmark(
                    env_=gym.BaseEnv(game="asterix", use_minimal_action_set=use_minimal_action_set_)),
                "breakout": PLBenchmark(
                    env_=gym.BaseEnv(game="breakout", use_minimal_action_set=use_minimal_action_set_)),
                "freeway": PLBenchmark(
                    env_=gym.BaseEnv(game="freeway", use_minimal_action_set=use_minimal_action_set_)),
                "seaquest": PLBenchmark(
                    env_=gym.BaseEnv(game="seaquest", use_minimal_action_set=use_minimal_action_set_)),
                "space_invaders": PLBenchmark(
                    env_=gym.BaseEnv(game="space_invaders", use_minimal_action_set=use_minimal_action_set_))
            }


class PLRegressor(RegressorMixin):

    def __init__(self, representation_,
                 config_,
                 hyperparameters_,
                 functions_,
                 terminals_,
                 num_episodes_=20):

        self.representation = representation_
        self.functions = functions_
        self.terminals = terminals_
        self.fitted_ = False
        self.config = config_
        self.hyperparameters = hyperparameters_
        self.num_episodes = num_episodes_

        if self.representation == "GE":
            self.arguments = [f"{t.name}" for t in self.terminals]
            self.grammar = self._make_default_grammar(self.functions, self.arguments, self.config.num_outputs)
        else:
            self.arguments = self.terminals
            self.grammar = None

    def _make_default_grammar(self, functions, arguments, num_outputs):
        # Ensure grammar uses uppercase function names matching Function objects
        return {
            "<expr>": ["[" + ', '.join([f"<lexpr>" for _ in range(num_outputs)]) + "]"],
            "<lexpr>": [f"{f.name.upper()}(<vexpr>, <vexpr>)" for f in functions if f.arity == 2]
                       + [f"{f.name.upper()}(<vexpr>)" for f in functions if f.arity == 1],
            "<vexpr>": [f"{f.name.upper()}(<vexpr>, <vexpr>)" for f in functions if f.arity == 2]
                       + [f"{f.name.upper()}(<vexpr>)" for f in functions if f.arity == 1]
                       + ["<var>"],
            "<var>": arguments
        }

    def fit(self, env, checkpoint=None):

        self.problem = PolicySearch(env=env, ideal_=self.config.ideal_fitness, minimizing_=False,
                                    num_episodes_=self.num_episodes)

        self.model = util.get_model(self.representation, self.functions, self.arguments,
                                    self.hyperparameters, self.config, self.grammar)

        if checkpoint is not None:
            self.model.resume(checkpoint, self.problem)

        self.program = self.model.evolve(self.problem)
        self.fitted_ = True

    def is_valid(self):
        if not self.fitted_:
            raise ValueError("Model not fitted")
        if not self.representation == "GE":
            raise ValueError("Method only works for GE")
        return self.model.is_valid(self.program.genome)

    def evaluate(self, num_episodes=10):
        if not self.fitted_:
            raise ValueError("Model not fitted")
        return self.problem.evaluate(self.program.genome, self.model, num_episodes=num_episodes, wait_key=False)
