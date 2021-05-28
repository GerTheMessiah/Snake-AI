import torch as T
import torch.nn as nn
import numpy as np

from src.snakeAI.agents.ppo.actor_critic import ActorCritic


class Agent:
    def __init__(self, lr_actor=2e-4, lr_critic=0.8e-3, gamma=0.99, K_epochs=10, eps_clip=0.1, gpu=True):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.critic_loss_coef = 0.5
        self.entropy_coef = 0.0001
        self.device = T.device('cuda:0' if T.cuda.is_available() and gpu else 'cpu')
        self.model_id = 0
        self.mseLoss = nn.MSELoss()

        self.policy = ActorCritic(n_actions=3, lr_actor=lr_actor, lr_critic=lr_critic, device=self.device)
        self.policy_old = ActorCritic(n_actions=3, lr_actor=lr_actor, lr_critic=lr_critic, device=self.device)
        self.policy_old.load_state_dict(self.policy.state_dict(destination=None))

    def learn(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = T.tensor(rewards, dtype=T.float64).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_around_view = T.cat(memory.around_view, axis=0).to(self.device).detach()
        old_cat_obs = T.cat(memory.cat_obs, axis=0).to(self.device).detach()
        old_actions = T.from_numpy(np.array(memory.actions)).to(self.device).detach()
        old_logprobs = T.cat(memory.logprobs).to(self.device).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_around_view, old_cat_obs, old_actions)
            ratios = T.exp(logprobs - old_logprobs.detach())
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = T.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss_actor = -(T.min(surr1, surr2) + dist_entropy * self.entropy_coef)
            loss_actor = loss_actor.mean()

            loss_critic = self.mseLoss(state_values, rewards) * self.critic_loss_coef
            loss_critic = loss_critic.mean()

            self.policy.base_actor.optimizer.zero_grad()
            loss_actor.backward()
            self.policy.base_actor.optimizer.step()

            self.policy.base_critic.optimizer.zero_grad()
            loss_critic.backward()
            self.policy.base_critic.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict(destination=None))
        T.cuda.empty_cache()

    def load_model(self, state_dict):
        self.policy_old.load_state_dict(state_dict=state_dict)
        self.policy.load_state_dict(state_dict=state_dict)
