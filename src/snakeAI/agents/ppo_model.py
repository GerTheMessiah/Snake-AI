import copy

import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import TensorType
import torch as T


class CustomModel(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super(CustomModel, self).__init__(obs_space=obs_space, action_space=action_space, num_outputs=num_outputs,
                                          model_config=model_config, name=name)

        self.critic_out = None
        self.n_classes_av_net = 128
        self.scalar_in = 41

        self.av_net_actor = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_classes_av_net)
        )

        self.av_net_critic = copy.deepcopy(self.av_net_actor)

        self.ACTOR_TAIL = nn.Sequential(
            nn.Linear(self.n_classes_av_net + self.scalar_in, self.n_classes_av_net),
            nn.ReLU(),
            nn.Linear(self.n_classes_av_net, self.n_classes_av_net),
            nn.ReLU(),
            nn.Linear(self.n_classes_av_net, num_outputs)
        )

        self.CRITIC_TAIL = nn.Sequential(
            nn.Linear(self.n_classes_av_net + self.scalar_in, self.n_classes_av_net),
            nn.ReLU(),
            nn.Linear(self.n_classes_av_net, self.n_classes_av_net),
            nn.ReLU(),
            nn.Linear(self.n_classes_av_net, 1),
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens) -> TensorType:
        av, scalar_obs = input_dict["obs"]
        av_actor_out = self.av_net_actor(av)
        av_critic_out = self.av_net_critic(av)
        concat_obs_actor = T.cat((av_actor_out, scalar_obs), dim=-1)
        concat_obs_critic = T.cat((av_critic_out, scalar_obs), dim=-1)
        actor_out = self.ACTOR_TAIL(concat_obs_actor)
        self.critic_out = self.CRITIC_TAIL(concat_obs_critic)

        return actor_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self.critic_out is not None, "must call forward() first"
        return self.critic_out.view(-1)
