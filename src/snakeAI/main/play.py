from time import sleep

from src.snakeAI.agents.ppo.ppo import Agent as AgentPPO
from src.snakeAI.agents.dqn.dqn import Agent as AgentDQN
from src.snakeAI.gym_game.snake_env import SnakeEnv
from src.snakeAI.agents.common.utils import file_path, plot_learning_curve
from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import torch as T


def test_play_ppo(n_iterations, print_stats=True):
    scores_l, apple_scores, wins = [], [], []
    scores = 0
    agent = AgentPPO()


    agent.policy_old.load_state_dict(T.load(get_model_path_(True)))
    game = SnakeEnv()
    game.post_init(field_size=(5, 5), has_gui=True)
    for i in range(1, n_iterations + 1):
        around_view, cat_obs = game.reset()
        while not game.game.p.done:
            around_view, cat_obs, action, _ = agent.policy_old.act(around_view, cat_obs)

            around_view_new, cat_obs_new, reward, done, won = game.step(action)
            scores += reward

            if game.game.has_gui:
                game.render()

            around_view = around_view_new
            cat_obs = cat_obs_new
            sleep(0.025)

        if print_stats:

            apple_count = game.game.p.apple_count
            print(f"Score: {round(scores, 2)} || Apple_Counter: {apple_count} || won: {won}")
            print("\n")
        sleep(0.025)
        scores = 0


def test_play_dqn(n_iterations, print_stats=True, has_gui=False):
    scores = 0
    scores_l, apple_scores, wins = [], [], []
    agent = AgentDQN(lr=1e-3, n_actions=3, gamma=0.99, epsilon=1.0, batch_size=2**8, eps_end=0.01, eps_dec=2e-5, max_mem_size=2**13)
    agent.Q_eval.load_state_dict(T.load("C:\\Users\\Lorenz Mumm\\PycharmProjects\\PPO-Snake-AI\\models\\dqn_model"))
    game = SnakeEnv()
    game.post_init(field_size=(8, 8), has_gui=has_gui)
    for i in range(1, n_iterations + 1):
        around_view, cat_obs = game.reset()
        while not game.has_ended:
            around_view, cat_obs, action = agent.act(around_view, cat_obs)

            around_view_new, cat_obs_new, reward, done, won = game.step(action)
            scores += reward
            agent.mem.add(around_view, cat_obs, action, reward, done, around_view_new, cat_obs_new)
            agent.learn()
            if game.game.has_gui:
                game.render()
            around_view = around_view_new
            cat_obs = cat_obs_new
            #sleep(0.025)

        if print_stats:
            apple_count = game.game.p.apple_count
            print(f"Score: {round(scores, 2)} || Apple_Counter: {apple_count} || won: {won} || epsilon: {agent.epsilon}")
            # print("\n")
        #sleep(0.025)
        scores_l.append(scores)
        wins.append(won)
        apple_scores.append(apple_count)
        scores = 0
    path_ = "C:\\Users\\Lorenz Mumm\\PycharmProjects\\PPO-Snake-AI\\stats\\dqn_model_stat"
    plot_learning_curve([i + 1 for i in range(len(scores_l))], apple_scores, scores_l, path_)
    T.save(agent.Q_eval.state_dict(), "C:\\Users\\Lorenz Mumm\\PycharmProjects\\PPO-Snake-AI\\models\\dqn_model")


if __name__ == '__main__':
    test_play_dqn(10000, True, False)
