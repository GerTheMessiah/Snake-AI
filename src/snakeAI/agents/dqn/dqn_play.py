import torch as T
from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from time import sleep
from src.snakeAI.agents.common.utils import file_path
from src.snakeAI.agents.dqn.dqn import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def play_dqn(n_iterations, print_stats=True, has_gui=True):
    agent = Agent(lr=1e-3, n_actions=3, gamma=0.99, epsilon=1.0, batch_size=2 ** 8, eps_end=0.01, eps_dec=2e-5,
                  max_mem_size=2 ** 13)
    agent.load_model(file_path(dir=r"models\dqn_models", new_save=False, file_name="model"))
    game = SnakeEnv()
    game.post_init(field_size=(8, 8), has_gui=has_gui)
    for i in range(1, n_iterations + 1):
        around_view, cat_obs = game.reset()
        scores = 0
        while not game.has_ended:
            around_view, cat_obs, action = agent.act(around_view, cat_obs)

            around_view_new, cat_obs_new, reward, done, won = game.step(action)
            scores += reward

            if game.has_gui:
                game.render()

            around_view = around_view_new
            cat_obs = cat_obs_new

        apple_count = game.apple_count
        if print_stats:
            print(f"Score: {round(scores, 2)} ||"
                  f" Apple_Counter: {apple_count} ||"
                  f" won: {won} ||"
                  f" epsilon: {agent.epsilon}")
            print("\n")
        sleep(0.025)
        scores = 0


if __name__ == '__main__':
    play_dqn(100, print_stats=True, has_gui=True)
