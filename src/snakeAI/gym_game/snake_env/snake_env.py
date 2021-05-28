import sys

import gym
from src.snakeAI.gym_game.snake_env.snake_game_2d import SnakeGame


class SnakeEnv(gym.Env):
    def __init__(self):
        self.shape = (10, 10)
        self.game = SnakeGame((self.shape[0], self.shape[1]), False)
        self.has_gui = False

    def step(self, action):
        self.game.action(action=action)
        around_view, cat_obs = self.game.observe()
        reward = self.game.evaluate()
        done = self.game.is_done()
        return around_view, cat_obs, reward, done, self.game.max_snake_length == self.game.p.apple_count + 1  # has won

    def reset(self):
        del self.game
        self.game = SnakeGame((self.shape[0], self.shape[1]), self.has_gui)
        around_view, cat_obs = self.game.observe()
        return around_view, cat_obs

    def render(self, close=False):
        self.game.view()

    def close(self):
        sys.exit()

    def post_init(self, field_size, has_gui):
        self.shape = field_size
        del self.game
        self.has_gui = has_gui
        self.game = SnakeGame(self.shape, self.has_gui)

    @property
    def has_ended(self):
        return self.game.p.done

    @property
    def apple_count(self):
        return self.game.p.apple_count
