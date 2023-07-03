import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import TensorType


class CustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super(CustomModel, self).__init__(obs_space=obs_space, action_space=action_space, num_outputs=num_outputs, model_config=model_config, name=name)

        class Flatten(nn.Module):
            def __init__(self):
                super(Flatten, self).__init__()

            def forward(self, x):
                if len(x.shape) == 3:
                    return torch.flatten(x, start_dim=0, end_dim=-1)
                elif len(x.shape) == 4:
                    return torch.flatten(x, start_dim=1, end_dim=-1)
                else:
                    raise ValueError("wrong dim")

        self.critic_out = None

        self.matrix_net = nn.Sequential(
            nn.Conv2d(in_channels=obs_space.shape[0], out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.critic_net = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(800, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1, bias=False),
        )

        self.actor_net = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(800, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3, bias=False),
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens) -> TensorType:
        matrix_input = input_dict["obs"]
        out = self.matrix_net(matrix_input)
        policy = self.actor_net(out)
        self.critic_out = self.critic_net(out)
        return policy, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self.critic_out is not None, "must call forward() first"

        return self.critic_out.view(-1)
