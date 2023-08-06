from tqdm import tqdm

from ray.rllib.algorithms import Algorithm

from src.gymnasium_env.snake.snake_env import SnakeEnv
from src.agents.ppo_model import CustomModel
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env
from ray.rllib.models import ModelCatalog
from src.agents.custom_metrics import CustomMetrics

if __name__ == '__main__':
    ray.init(num_gpus=1)

    register_env("snake-v0", lambda config: SnakeEnv(config))
    ModelCatalog.register_custom_model("CNNSnakeModel", CustomModel)

    env_name = "snake-v0"

    ppo_config = PPOConfig()
    ppo_config = ppo_config.training(model={"custom_model": "CNNSnakeModel", "vf_share_layers": True},
                                     lr_schedule=[[0, 1.25e-4], [10_000_000, 1.00e-4], [15_000_000, 0.85e-4], [20_000_000, 0.75e-4]],
                                     use_critic=True,
                                     use_gae=False,
                                     lambda_=0.95,
                                     train_batch_size=2 ** 14,
                                     sgd_minibatch_size=2 ** 10,
                                     num_sgd_iter=4,
                                     shuffle_sequences=True,
                                     vf_loss_coeff=0.5,
                                     entropy_coeff=0.01,
                                     entropy_coeff_schedule=[[0, 0.0], [10_000_000, 0.0], [15_000_001, 0.001]],
                                     clip_param=0.2,
                                     lr=1.25e-4,
                                     gamma=0.95,
                                     _enable_learner_api=False)

    ppo_config = ppo_config.resources(num_gpus=1, num_cpus_per_worker=1)
    ppo_config = ppo_config.rollouts(batch_mode="complete_episodes", num_rollout_workers=8, num_envs_per_worker=2)
    ppo_config = ppo_config.environment(env=env_name, env_config={"shape": (10, 10)}, disable_env_checking=True, is_atari=False)
    ppo_config = ppo_config.framework(framework="torch")
    ppo_config = ppo_config.callbacks(CustomMetrics)
    ppo_config = ppo_config.checkpointing(export_native_model_files=True)
    ppo_config = ppo_config.exploration(explore=True, exploration_config={"type": "StochasticSampling"})

    algo: Algorithm = ppo_config.build()
    print()
    for _ in tqdm(range(2_000), colour="green"):
        algo.train()
    ray.shutdown()
