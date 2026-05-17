"""
Benchmark representation module for policy learning benchmarks.

 - ALEArgs: Arguments for the Atari preprocessing
 - PLBenchmark: Class to represent a PL benchmark, performs the preprocessing for
                ALE and non-ALE environments with a wrapper
"""

from dataclasses import dataclass

import gymnasium as gym

from src.benchmark.benchmark import Benchmark
from gymnasium.wrappers import FlattenObservation
from functools import reduce

from src.gp.tinyverse import Var


@dataclass
class PLArgs:
    max_steps: int
    max_episode_steps: int
    difficulty: int
    flatten_obs: bool

@dataclass
class ALEArgs(PLArgs):
    """
    Arguments for the preprocessing of
    Gymnasium Arcade Learning Environment (A.L.E).
    """
    noop_max: int
    frame_skip: int
    screen_size: int
    grayscale_obs: int
    terminal_on_life_loss: int
    scale_obs: int
    frame_stack: int
    frames_per_step: int
    full_action_space: bool
    repeat_action_probability: float


@dataclass
class MinAtarArgs(PLArgs):
    """
    Arguments for the MinAtar benchmarks.
    """
    use_minimal_action_set: bool


class PLBenchmark(Benchmark):
    """
    Class for representing policy learning benchmarks as provided
    by Gymnasium. This class currently supports the Box2D, Classic control and ALE environments.
    The ALE environments are ony supported with RGB or grayscale observation space.
    """

    def __init__(
            self, env_: gym.Env, args_: PLArgs
    ):
        self.env = env_
        self.wrapped_env = env_
        self.ale = True if isinstance(args_, ALEArgs) else False
        self.flatten_obs = args_.flatten_obs
        self.generate(args=args_)

    def generate(self, args: any):
        """
        Wraps the Gym environment either into a flat wrapper or into a
        Atari preprocessing wrapper.

        The observation space is flatten is its declared true at time
        of instantiation.

        :param args: Atari arguments for processing
        :return: preprocessed environment
        """
        if self.ale:
            if args is not None:
                self.wrapped_env = gym.wrappers.AtariPreprocessing(
                    self.env,
                    noop_max=args.noop_max,
                    frame_skip=args.frame_skip,
                    screen_size=args.screen_size,
                    grayscale_obs=args.grayscale_obs,
                    terminal_on_life_loss=args.terminal_on_life_loss,
                    scale_obs=args.scale_obs,
                )

                if args.frame_stack > 0:
                    self.wrapped_env = gym.wrappers.FrameStackObservation(self.wrapped_env, 4)

            else:
                self.wrapped_env = gym.wrappers.AtariPreprocessing(self.env)
        else:
            self.wrapped_env = self.env

        if self.flatten_obs:
            self.wrapped_env = FlattenObservation(self.wrapped_env)

    def len_observation_space(self):
        """
        Returns the size of the observation space.
        """
        return reduce(lambda x, y: x * y, self.wrapped_env.observation_space.shape)

    def len_action_space(self):
        """
        Returns the size of action space.
        """
        return self.env.action_space.n

    def gen_terminals(self):
        """
        Generates and returns the terminals to be used by
        the GP model.
        """
        return [Var(i, "Inp" + str(i)) for i in range(self.len_observation_space())]
