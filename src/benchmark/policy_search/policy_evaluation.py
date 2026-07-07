"""
Benchmark representation module for policy search.

- GPAgent: Class is used to represent a agent that is equipped with a candidate policy

"""

import gymnasium as gym
from src.gp.tinyverse import GPModel
from gymnasium.wrappers import FlattenObservation
import statistics


class GPAgent:
    """
    Agent class that is placed in reinforcement learning environments
    """

    def __init__(self, env_: gym.Env, flatten_obs=True):
        """
        :param env_: Environment
        :param flatten_obs: Option to flatten the obversation
        """
        self.env = env_
        if flatten_obs:
            self.wrapped_env = FlattenObservation(self.env)



    def evaluate_policy(self, policy, model, num_episodes=100, wait_key=False):
        """
        Evaluates a policy in an environment with the selected number of episodes.

        :param policy: Candidate policy from the GP model
        :param model: The GP model with which the policy is evolved
        :param num_episodes: Number of episodes
        :param wait_key: Wait key option to see the end result of an episode

        :return: Mean of cumulative rewards
        """
        rewards = []
        step_cnt = 0
        for episode \
                in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            cumulative_reward = 0
            while not done:
                if self.wrapped_env is not None:
                    obs_ = self.wrapped_env.observation(obs)
                else:
                    obs_ = obs
                action = self.get_action(policy, model, obs_)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                obs = next_obs
                step_cnt += 1
                cumulative_reward += reward
            if wait_key:
                input("Press Enter to continue...")
            rewards.append(cumulative_reward)
        return statistics.mean(rewards), step_cnt

    def get_action(self, policy: list[int], model: GPModel, obs):
        """
        Predicts the action of an agent equipped with a candidate policy based
        on the given observation.

        :param policy: candidate policy
        :param model: GP model
        :param obs: observation
        :return: predicted agent that is being performed by the agent
        """
        prediction = model.predict(policy, obs)
        maximum = max(prediction)
        return prediction.index(maximum)
