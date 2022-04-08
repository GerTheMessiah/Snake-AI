import os
from os import environ
from src.gym_snake.envs.snake_env import SnakeEnv
from src.snakeAI.agents import CustomModel

environ['TUNE_DISABLE_AUTO_CALLBACK_SYNCER'] = '1'
import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents import ppo
from ray.rllib.utils.typing import TrainerConfigDict
from ray.tune import tune, register_env
from ray.rllib.models import ModelCatalog
from src.snakeAI.agents.custom_metrics import CustomMetrics
import argparse

parser = argparse.ArgumentParser()
project_root = os.path.dirname(os.path.dirname(__file__))
output_path = os.path.join(project_root, 'src\\resources\\models')
# --------------------------ray-------------------------------#
parser.add_argument("--iterations", type=int, default=2000,
                    help="The number of iterations to train the model")
parser.add_argument("--num_gpus", type=int, default=1,
                    help="The number of gpus.")
parser.add_argument("--dashboard", type=bool, default=True,
                    help="Use Tensorboard dashboard.")
parser.add_argument("--path", type=str, default=output_path,
                    help="Use Tensorboard dashboard.")
parser.add_argument("--train_batch_size", type=int, default=1024,
                    help="Batch size of all envs together.")
parser.add_argument("--sgd_minibatch_size", type=int, default=128,
                    help="Mini batch size for training.")
parser.add_argument("--num_sgd_iter", type=int, default=10,
                    help="Training epochs")
parser.add_argument("--rollout_fragment_length", type=int, default=256,
                    help="Minimum length of PPO Memory per worker")
parser.add_argument("--checkpoint_freq", type=int, default=100,
                    help="Number of training steps before the model gets save.")

# -----------------------------Environment----------------------------------#
parser.add_argument("--env", type=str, default="snake-v0",
                    help="Custom Snake Environment")
parser.add_argument("--env_shape", type=tuple, default=(8, 8),
                    help="Size of the square environment.")
parser.add_argument("--env_has_gui", type=bool, default=False,
                    help="Use GUI.")

# ---------------------------PPO-------------------------------#
parser.add_argument("--num_worker", type=int, default=4,
                    help="Number of the active game data creating workers.")
parser.add_argument("--entropy_coeff", type=float, default=1e-3,
                    help="PPO entropy-coefficient.")
parser.add_argument("--lr", type=float, default=0.2e-4,
                    help="PPO learning-rate.")
parser.add_argument("--vf_loss_coeff", type=float, default=0.5,
                    help="Value-loss-coefficient of the ppo.")
parser.add_argument("--lambdaGAE", type=float, default=0.95,
                    help="PPO GAE lambda coefficient.")
parser.add_argument("--clip_param", type=float, default=0.2,
                    help="PPO clip parameter (0.1 to 0.3).")
parser.add_argument("--explore", type=bool, default=True,
                    help="Forces the PPO to choice random actions " +
                         "in regard to the computed probability distribution.")

register_env("snake-v0", SnakeEnv)
ModelCatalog.register_custom_model("CustomModelCNN", CustomModel)

if __name__ == '__main__':
    args = parser.parse_args()
    ray.init(num_gpus=args.num_gpus, include_dashboard=args.dashboard)
    env_name = "snake-v0"
    r = dict(ppo.DEFAULT_CONFIG.copy())
    r.update({
        "env": env_name,
        "env_config": {"shape": args.env_shape, "has_gui": args.env_has_gui},
        "log_level": "ERROR",
        "framework": "torch",
        "num_workers": args.num_worker,
        "num_gpus": args.num_gpus,
        "model": {"custom_model": "CustomModelCNN"},
        "entropy_coeff": args.entropy_coeff,
        "batch_mode": "complete_episodes",
        "lr": args.lr,
        "kl_target": 0,
        "kl_coeff": 0,
        "vf_loss_coeff": args.vf_loss_coeff,
        "lambda": args.lambdaGAE,
        "clip_param": args.clip_param,
        "explore": args.explore,
        "exploration_config": {
            "type": "StochasticSampling"
        },
        "train_batch_size": args.train_batch_size,
        "sgd_minibatch_size": args.sgd_minibatch_size,
        "num_sgd_iter": args.num_sgd_iter,
        "clip_rewards": False,
        "preprocessor_pref": None,
        "normalize_actions": False,
        "rollout_fragment_length": args.rollout_fragment_length,
        "callbacks": CustomMetrics,
    })
    config = TrainerConfigDict(r)

    tune.run(PPOTrainer, name="PPOStandardCNN", stop={"training_iteration": args.iterations},
             config=config, checkpoint_at_end=True, checkpoint_freq=args.checkpoint_freq,
             keep_checkpoints_num=3, checkpoint_score_attr="episode_reward_mean",
             local_dir=args.path, restore="your Tune restore path", resume=True)
