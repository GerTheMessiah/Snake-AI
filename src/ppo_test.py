import time

import gym
import ray
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from src.gym_snake.envs.snake_env import SnakeEnv
from src.snakeAI.agents.ppo_model import CustomModel

if __name__ == '__main__':

    ray.init(num_gpus=1, include_dashboard=True, )
    env_name = "snake-v0"
    register_env(env_name, SnakeEnv)
    ModelCatalog.register_custom_model("CustomModel", CustomModel)
    env_name = "snake-v0"
    r = dict(ppo.DEFAULT_CONFIG.copy())
    r.update(dict(env=env_name, env_config={"shape": (8, 8), "has_gui": False}, log_level="ERROR", framework="torch",
                  num_workers=1, num_gpus=1, model={"custom_model": "CustomModel"}, use_gae=True,
                  batch_mode="complete_episodes", lr=3.5e-3, gamma=0.95, clip_param=0.2, kl_coeff=0.0,
                  shuffle_sequences=True, sgd_minibatch_size=128, num_sgd_iter=10, vf_loss_coeff=0.5,
                  entropy_coeff=0.01,
                  train_batch_size=1000, rollout_fragment_length=500, vf_clip_param=0.0, evaluation_interval=2,
                  evaluation_num_episodes=20))

    agent = PPOTrainer(config=r)
    path = "path_to_checkpoint\\checkpoint-xxx"
    agent.restore(path)

    env = gym.make(env_name, config={'shape': (8, 8), 'has_gui': True})
    obs = env.reset()
    while True:
        action = agent.compute_action(obs)
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(0.1)
        if done:
            obs = env.reset()
