import numpy as np
import math


class Reward:
    def __init__(self, snake_game):
        self.has_grown = False
        self.snakeGame = snake_game
        self.snake_dist_old = 0
        self.snake_len_old = 1

    """
    This method is returning the standard reward.
    """
    @property
    def standard_reward(self):
        if self.snakeGame.p.snake_len == self.snakeGame.max_snake_length and self.snakeGame.p.is_terminal:
            return 100
        elif self.snakeGame.p.snake_len != self.snakeGame.max_snake_length and self.snakeGame.p.is_terminal:
            return -10
        elif self.has_grown:
            return 2.5
        else:
            return -0.01
    """
    This method is returning the optimized reward.
    """
    @property
    def optimized_reward(self):
        if len(self.snakeGame.p.tail) == self.snakeGame.max_snake_length and self.snakeGame.p.is_terminal:
            return 100
        elif len(self.snakeGame.p.tail) != self.snakeGame.max_snake_length and self.snakeGame.p.is_terminal:
            return -10
        elif self.has_grown:
            return 2.5 * ((1 / 63) * self.snakeGame.p.snake_len + 1)
        else:
            delta_reward = 0
            reward = -0.01
            len_ = self.snakeGame.p.snake_len
            dist = np.linalg.norm(self.snakeGame.p.pos - np.array(self.snakeGame.apple))
            if self.snake_len_old > 1:
                delta_reward = math.log((self.snake_len_old + self.snake_dist_old) /
                                        (self.snake_len_old + dist), self.snake_len_old)
            self.snake_dist_old = dist
            self.snake_len_old = len_
            return min(-0.001, max(reward + delta_reward, -0.02))
