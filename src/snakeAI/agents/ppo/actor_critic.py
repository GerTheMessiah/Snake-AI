import torch as T
import torch.nn as nn
from torch.distributions import Categorical

from src.snakeAI.agents.common.base import BaseNet


class ActorCritic(nn.Module):
    def __init__(self, n_actions, lr_actor=1e-4, lr_critic=1e-4, device="cpu"):
        super(ActorCritic, self).__init__()
        T.manual_seed(10)
        self.device = device

        self.base_actor = BaseNet(n_actions, 'actor', lr=lr_actor)
        self.base_critic = BaseNet(1, 'critic', lr=lr_critic)
        self.to(self.device)

    def forward(self, around_view, cat_obs):
        actor = self.base_actor(around_view, cat_obs)
        critic = self.base_critic(around_view, cat_obs)
        return actor, critic

    @T.no_grad()
    def act(self, around_view, cat_obs):
        around_view_ = T.from_numpy(around_view).to(self.device)
        cat_obs_ = T.from_numpy(cat_obs).to(self.device)
        action_probs = self.base_actor(around_view_, cat_obs_)
        dist = Categorical(action_probs)
        action = dist.sample()
        return around_view_, cat_obs_, action.item(), dist.log_prob(action)

    def evaluate(self, around_view, cat_obs, action):
        policy, value = self.forward(around_view, cat_obs)
        dist = Categorical(policy)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, T.squeeze(value), dist_entropy
