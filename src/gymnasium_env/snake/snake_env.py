import random
from typing import Any
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box

from src.gymnasium_env.snake.observation import Observation, _on_playground


def manhattan_distance(a, b):
    return (abs(a[0] - b[0])) + (abs(a[1] - b[1]))


class SnakeEnv(gym.Env):
    def __init__(self, config: dict):
        super(SnakeEnv, self).__init__()
        self.shape = config["shape"]

        self.action_space = Discrete(4)
        self.observation_space = Box(0.0, 1.0, (1, 11, self.shape[0], self.shape[1]), dtype=np.float32)
        self.reward_range = (-10., 20.)

        self.tail = [(self.shape[0] // 2, self.shape[1] // 2)]
        self.direction = 0

        self.observation = Observation(obs_shape=self.shape)

        self.hunger = 0
        self.num_steps = 0

        self.grown = False

        self.game_positions = set((i, j) for i in range(self.shape[0]) for j in range(self.shape[1]))

        self.apple = self.make_apple()

        self.previous_distance = manhattan_distance(self.tail[0], self.apple)

    def step(self, action: int) -> (dict, float, bool, bool, dict[str, Any]):
        if self.hunger == self.shape[0] * self.shape[1]:
            self.grown = False
            return self.observation.create_view(self.apple, self.tail, self.direction, self.hunger), self.evaluate(False, True), False, True, self.make_info()

        self.hunger += 1
        self.num_steps += 1

        self.direction = action

        head = [self.head[0], self.head[1]]
        head[self.direction % 2] += -1 if self.direction % 3 == 0 else 1
        self.tail.insert(0, (head[0], head[1]))

        # If snake ran out of playground.
        if not _on_playground(self.head[0], self.head[1], self.shape):
            self.grown = False
            self.tail.pop()
            return self.observation.create_view(self.apple, self.tail, self.direction, self.hunger), self.evaluate(True, False), True, False, self.make_info()

        # If snake ran into itself.
        if self.tail[0] in self.tail[1:-1]:
            self.tail.pop()
            self.grown = False
            return self.observation.create_view(self.apple, self.tail, self.direction, self.hunger), self.evaluate(True, False), True, False, self.make_info()

        if self.head == self.apple:
            self.hunger = 0
            self.grown = True

            if self.won:
                return self.observation.create_view(self.apple, self.tail, self.direction, self.hunger), self.evaluate(True, False), True, False, self.make_info()

            self.apple = self.make_apple()
            return self.observation.create_view(self.apple, self.tail, self.direction, self.hunger), self.evaluate(False, False), False, False, self.make_info()
        else:
            self.tail.pop()
            self.grown = False
            return self.observation.create_view(self.apple, self.tail, self.direction, self.hunger), self.evaluate(False, False), False, False, self.make_info()

    def reset(self, *, seed=None, options=None):
        self.tail.clear()
        self.tail = [(self.shape[0] // 2, self.shape[1] // 2)]
        self.direction = 0

        self.apple = self.make_apple()
        self.hunger = 0
        self.num_steps = 0
        self.previous_distance = 0

        self.grown = False
        self.observation.previous_obs = None

        return self.observation.create_view(self.apple, self.tail, self.direction, self.hunger), self.make_info()

    def make_info(self):
        return {"won": self.won, "apple_count": self.apple_count, "steps": self.num_steps}

    def render(self):
        return

    def close(self):
        return

    def make_apple(self):
        game_postions = self.game_positions - set(self.tail)

        if not game_postions:
            return None

        return random.sample(game_postions, 1)[0]

    def evaluate(self, terminal: bool, truncated: bool):
        if self.won:
            return 5.0

        if terminal:
            return -1.0

        if truncated:
            return -0.25

        dist = manhattan_distance(self.tail[0], self.apple)

        if self.grown:
            self.previous_distance = dist
            return 0.25

        if self.previous_distance > dist:
            self.previous_distance = dist
            return 0.005

        elif self.previous_distance < dist:
            self.previous_distance = dist
            return -0.0025

        else:
            return -0.0001

    @property
    def apple_count(self):
        return len(self.tail) - 1

    @property
    def snake_length(self) -> int:
        return len(self.tail)

    @property
    def head(self):
        return self.tail[0]

    @property
    def max_snake_length(self) -> int:
        return self.shape[0] * self.shape[1]

    @property
    def won(self):
        return len(self.tail) == self.shape[0] * self.shape[1]
