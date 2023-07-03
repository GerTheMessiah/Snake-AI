from typing import Dict, Optional, Tuple, Union
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import RolloutWorker, Episode
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.typing import PolicyID


class CustomMetrics(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode, env_index: Optional[int] = None, **kwargs):
        episode.custom_metrics = {}

    def on_episode_step(self, *, worker: "RolloutWorker", base_env: BaseEnv, policies: Optional[Dict[PolicyID, Policy]] = None, episode: Union[Episode, EpisodeV2],
                        env_index: Optional[int] = None, **kwargs,):
        # Implementation not required.
        pass

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: Episode, env_index: Optional[int] = None, **kwargs):
        episode.custom_metrics["won"] = worker.env.won
        episode.custom_metrics["apple_count"] = worker.env.apple_count

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        # Implementation not required.
        pass

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        # Implementation not required.
        pass

    def on_postprocess_trajectory(self, *, worker: RolloutWorker, episode: Episode, agent_id: str, policy_id: str, policies: Dict[str, Policy], postprocessed_batch: SampleBatch,
                                  original_batches: Dict[str, Tuple[Policy, SampleBatch]], **kwargs):
        # Implementation not required.
        pass
