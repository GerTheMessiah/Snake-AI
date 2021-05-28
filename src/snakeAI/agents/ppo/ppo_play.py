from os import environ
from time import sleep
import torch as T

from src.snakeAI.agents.common.utils import file_path

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from src.snakeAI.agents.ppo.ppo import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def test_play(n_iterations, print_stats=True):
    agent = Agent()
    agent.load_model(T.load(file_path(dir=r'models\ppo_models', new_save=False, file_name="model")))
    game = SnakeEnv()
    game.post_init(field_size=(8, 8), has_gui=True)
    for i in range(1, n_iterations + 1):
        around_view, cat_obs = game.reset()
        scores = 0
        while not game.has_ended:
            around_view, cat_obs, action, _ = agent.policy_old.act(around_view, cat_obs)
            around_view_new, cat_obs_new, reward, done, won = game.step(action)
            scores += reward

            if game.has_gui:
                game.render()

            around_view = around_view_new
            cat_obs = cat_obs_new
            sleep(0.255)
        apple_count = game.apple_count
        if print_stats:
            print(f"Score: {round(scores, 2)} || Apple_Counter: {apple_count} || won: {won}")
            print("\n")
        sleep(0.255)


if __name__ == '__main__':
    test_play(100, True)
