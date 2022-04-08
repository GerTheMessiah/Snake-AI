from typing import Dict
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks


class CustomMetrics(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode, **kwargs):
        episode.custom_metrics = {}

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv, episode: MultiAgentEpisode, **kwargs):
        # Implementation not required.
        pass

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[str, Policy], episode: MultiAgentEpisode, **kwargs):
        episode.custom_metrics["won"] = worker.env.has_won
        episode.custom_metrics["apples"] = worker.env.apple_count

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        # Implementation not required.
        pass

    def on_train_result(self, trainer, result: dict, **kwargs):
        # Implementation not required.
        pass

    def on_postprocess_trajectory(self, worker: RolloutWorker, episode: MultiAgentEpisode, agent_id: str, policy_id: str, policies: Dict[str, Policy], postprocessed_batch: SampleBatch, original_batches: Dict[str, SampleBatch], **kwargs):
        # Implementation not required.
        pass
