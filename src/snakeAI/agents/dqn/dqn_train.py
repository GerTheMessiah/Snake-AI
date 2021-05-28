import datetime
from statistics import mean
from time import time
from datetime import timedelta
from os import environ

from pandas import DataFrame

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

from src.snakeAI.agents.common.utils import plot_learning_curve, file_path, print_progress
from src.snakeAI.agents.dqn.dqn import Agent
from src.snakeAI.gym_game.snake_env import SnakeEnv


def train_dqn(N_ITERATIONS, print_stats=False, has_gui=False):
    LR_ACTOR, score, BOARD_SIZE = 1e-3, 0, (8, 8)
    start_time = time()
    scores, apples, wins, dtime = [], [], [], []
    agent = Agent(lr=LR_ACTOR, n_actions=3, gamma=0.99, epsilon=1.0, batch_size=2 ** 8, eps_end=0.01, eps_dec=2e-5,
                  max_mem_size=2 ** 13)
    agent.load_model(file_path(dir=r"models\dqn_models", new_save=False, file_name="model"))
    game = SnakeEnv()
    game.post_init(field_size=(8, 8), has_gui=has_gui)
    iter_time = time()
    for i in range(1, N_ITERATIONS + 1):
        around_view, cat_obs = game.reset()
        while not game.has_ended:
            around_view, cat_obs, action = agent.act(around_view, cat_obs)

            around_view_new, cat_obs_new, reward, done, won = game.step(action)
            score += reward

            agent.mem.add(around_view, cat_obs, action, reward, done, around_view_new, cat_obs_new)
            agent.learn()

            if game.has_gui:
                game.render()

            around_view = around_view_new
            cat_obs = cat_obs_new

        apple_count = game.apple_count
        if print_stats:
            print(f"Score: {round(score, 2)} || Apple_Counter: {apple_count} || won: {won} || epsilon: {agent.epsilon}")
            print("\n")

        scores.append(score)
        wins.append(won)
        apples.append(apple_count)
        dtime.append(datetime.datetime.now())
        score = 0

        t = time()

        time_step = str(timedelta(seconds=t - iter_time) * (N_ITERATIONS - i))
        passed_time = str(timedelta(seconds=t - start_time))
        iter_time = time()
        suffix_1 = f"Passed Time: {passed_time} | Remaining Time: {time_step}"
        suffix_2 = f" | Food_avg: {round(mean(apples[-5:]), 2)} | Score_avg: {round(mean(scores[-5:]), 2)}"
        suffix_3 = f" | Wins:{int(mean(wins[-5:]))}"
        print_progress(i, N_ITERATIONS, suffix=suffix_1 + suffix_2 + suffix_3)

    path_ = file_path(dir=r"stats\dqn_stats", new_save=True, file_name="stats")
    plot_learning_curve([i + 1 for i in range(len(scores))], apples, scores, path_)
    agent.save_model(new_save=True)
    columns = ["datetime", "apples", "scores", "wins", "lr_actor", "lr_critic",
               "board_size_x", "board_size_y"]
    c = {"datetime": dtime, "apples": apples, "scores": scores, "wins": wins,
         "lr_actor": LR_ACTOR, "lr_critic": None, "board_size_x": BOARD_SIZE[0], "board_size_y": BOARD_SIZE[1]}
    df = DataFrame(c, columns=columns)
    df.to_csv(file_path(r'csv\dqn_csv', new_save=True, file_name='train'))
    print(df)


if __name__ == '__main__':
    train_dqn(N_ITERATIONS=1000, print_stats=False, has_gui=False)
