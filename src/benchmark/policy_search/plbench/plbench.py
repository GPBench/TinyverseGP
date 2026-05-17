from dataclasses import dataclass
from sklearn.base import RegressorMixin
from src.benchmark.benchmark import Benchmark
from minatar import gym as gym_ma
import ale_py
import gymnasium as gym
from src.benchmark.policy_search.pl_benchmark import PLBenchmark, ALEArgs, MinAtarArgs
from src.gp import util
from src.gp.problem import PolicySearch


@dataclass
class PLBenchConfig:
    ale_args: ALEArgs
    minatar_args: MinAtarArgs


class PLBench(Benchmark):
    """
    Benchmark class that is used for benchmarking GP in policy learning. It is
    divided into two subclasses of benchmark problems:
        - MinAtar: An Atari-Inspired Testbed for Thorough and Reproducible Reinforcement Learning Experiments)
            - https://github.com/kenjyoung/MinAtar
            - https://arxiv.org/abs/1903.03176
        - Atari5: Distilling the Arcade Learning Environment down to Five Games
            - https://arxiv.org/abs/2210.02019
    """

    def __init__(self, ale_args_: ALEArgs, minatar_args_: MinAtarArgs):
        self.ale_args = ale_args_
        self.minatar_args = minatar_args_
        gym.register_envs()
        self.generate(args=None)

    def generate(self, args: any):
        self.benchmark = {"minatar": PLBench.MinAtar().problems,
                          "atari_5": PLBench.AtariFive(self.ale_args).problems}

    class MinAtar:
        def __init__(self, args: MinAtarArgs):
            use_minimal_action_set = args.use_minimal_action_set
            max_episodes_steps = args.max_episode_steps

            gym_ma.register_envs()

            self.problems = {
                "asterix": PLBenchmark(
                    env_=gym.make(id='MinAtar/Asterix-v1', max_episode_steps=max_episodes_steps,
                                  use_minimal_action_set=use_minimal_action_set, render_mode="rgb_array"),
                    args_=args),
                "breakout": PLBenchmark(
                    env_=gym.make(id='MinAtar/Breakout-v1', max_episode_steps=max_episodes_steps,
                                  use_minimal_action_set=use_minimal_action_set, render_mode="rgb_array"),
                    args_=args),
                "freeway": PLBenchmark(
                    env_=gym.make(id='MinAtar/Freeway-v1', max_episode_steps=max_episodes_steps,
                                  use_minimal_action_set=use_minimal_action_set, render_mode="rgb_array"),
                    args_=args),
                "seaquest": PLBenchmark(
                    env_=gym.make(id='MinAtar/Seaquest-v1', max_episode_steps=max_episodes_steps,
                                  use_minimal_action_set=use_minimal_action_set, render_mode="rgb_array"),
                    args_=args),
                "space_invaders": PLBenchmark(
                    env_=gym.make(id='MinAtar/SpaceInvaders-v1', max_episode_steps=max_episodes_steps,
                                  use_minimal_action_set=use_minimal_action_set, render_mode="rgb_array"),
                    args_=args),
            }

    class AtariFive:
        def __init__(self, args: ALEArgs):
            self.problems = {
                "battle_zone": PLBenchmark(env_=gym.make("ALE/BattleZone-v5"), args_=args),
                "double_dunk": PLBenchmark(env_=gym.make("ALE/DoubleDunk-v5"), args_=args),
                "name_this_game": PLBenchmark(env_=gym.make("ALE/NameThisGame-v5"), args_=args),
                "phoenix": PLBenchmark(env_=gym.make("ALE/Phoenix-v5"), args_=args),
                "qbert": PLBenchmark(env_=gym.make("ALE/Qbert-v5"), args_=args),
            }

        def get_ale_id(self, id):
            return self.problems[id].env.spec.id


class PLRegressor(RegressorMixin):
    """
    Regressor class for benchmarking policy learning
    """

    def __init__(self, representation_,
                 config_,
                 hyperparameters_,
                 functions_,
                 terminals_,
                 num_episodes_=100):

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
            "<start>": ["<expr_list>"],

            "<expr_list>": [
            "<expr>",
            "<expr>,<expr_list>"
            ],

            "<expr>": [f"{f.name.upper()}(<expr>, <expr>, <expr>)" for f in functions if f.arity == 3]
                       + [f"{f.name.upper()}(<expr>, <expr>)" for f in functions if f.arity == 2]
                       + [f"{f.name.upper()}(<expr>)" for f in functions if f.arity == 1]
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
