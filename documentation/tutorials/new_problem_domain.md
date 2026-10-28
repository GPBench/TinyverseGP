# Adding a problem domain module

This tutorial demonstrates how to add a new problem domain module based on the `PolicySearch` class.

New problem domains are to be added as new classes to `src/gp/problem.py` which includes an abstract class `Problem` which each problem class inherits from. The following methods have already been implemented:

#### `is_ideal(self, fitness)`
Returns `True` for an individual with optimal fitness as configured in `ideal_fitness` in `GPConfig`.

#### `is_better(self, fitness1, fitness2)`
Simple comparison method that returns `True` if `fitness1` is better than `fitness2`. 

The new problem class should implement the following methods:

#### `evaluate(self, genome, model: GPModel)`
This method evaluates a genome. The exact implementation depends on the problem domain and GP representation used.

```python title="Example"
def evaluate(
            self, genome, model: GPModel, num_episodes: int = None, wait_key: bool = False
    ) -> float:
        if num_episodes is None:
            num_episodes = self.num_episodes
        reward, num_steps = self.agent.evaluate_policy(genome, model, num_episodes, wait_key)

        if self.frames_per_step is not None:
            self.step_cnt += num_steps * self.frames_per_step
        else:
            self.step_cnt += num_steps
        self.log_evaluation(reward, genome)
        return reward
```

#### `is_stop(self)`
`is_stop` can be used to implement optional stopping conditions. For Policy Learning, one could add a maximum number of steps:

```python title="Example"
def is_stop(self):
        return self.step_cnt >= self.max_steps
```

### Example: `PolicySearch`

For a complete example, take a look at the complete `PolicySearch` class:

```python
class PolicySearch(Problem):
    """
    Representation of the policy search problem where the fitness is calculated by the
    average reward of the policy.

    An instance of the GPAgent class is used to evaluate a candidate policy within the respective
    environment.
    """

    agent: GPAgent
    num_episodes: int
    max_steps: int
    step_cnt: int
    frames_per_step = int

    def __init__(
            self, env: gym.Env, ideal_: float, minimizing_: bool, num_episodes_: int = 100,
            frames_per_step_: int = 1,
            max_steps_: float = 2e6, fid: int = 1, name: str = "None", logger: ioh.logger.AbstractLogger = None):
        super().__init__(fid=fid, name=name, minimizing=minimizing_, logger=logger)
        self.agent = GPAgent(env)
        self.ideal = ideal_
        self.minimizing = minimizing_
        self.num_episodes = num_episodes_
        self.frames_per_step = frames_per_step_
        self.max_steps = int(max_steps_)
        self.step_cnt = 0

    def is_stop(self):
        return self.step_cnt >= self.max_steps

    def evaluate(
            self, genome, model: GPModel, num_episodes: int = None, wait_key: bool = False
    ) -> float:
        if num_episodes is None:
            num_episodes = self.num_episodes
        reward, num_steps = self.agent.evaluate_policy(genome, model, num_episodes, wait_key)

        if self.frames_per_step is not None:
            self.step_cnt += num_steps * self.frames_per_step
        else:
            self.step_cnt += num_steps
        self.log_evaluation(reward, genome)
        return reward
```
