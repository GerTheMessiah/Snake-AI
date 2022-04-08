import gym
from ray.rllib.env import EnvContext

from src.gym_snake.envs.snake_env.snake_game import SnakeGame
from gym import spaces
import numpy as np


class SnakeEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.shape = config["shape"]
        self.has_gui = config["has_gui"]
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((spaces.Box(low=-2.0, high=2.0, shape=(6, 13, 13), dtype=np.float64),
                                               spaces.Box(low=-2.0, high=2.0, shape=(41,), dtype=np.float64)))

        self.game = SnakeGame(self.shape, self.has_gui)

    def step(self, action: int):
        self.game.action(action=action)
        around_view, scalar_obs = self.game.observe()
        reward = self.game.evaluate()
        is_terminal = self.game.is_terminal
        return [around_view, scalar_obs], reward, is_terminal, {
            "won": self.game.max_snake_length == self.game.p.apple_count + 1, "apples": self.game.p.apple_count}  # has won

    def reset(self, seed=None, return_info=None, options=None):
        self.game.reset_snake_game(new_shape=self.shape)
        around_view, scalar_obs = self.game.observe()
        return [around_view, scalar_obs]

    def render(self, mode="human"):
        self.game.view()

    def close(self):
        pass

    @property
    def has_ended(self):
        return self.game.p.is_terminal

    @property
    def apple_count(self):
        return self.game.p.apple_count

    @property
    def has_won(self):
        return self.game.p.apple_count + 1 == self.game.max_snake_length
