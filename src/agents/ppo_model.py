import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import TensorType


class CustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, *args, **kwargs):
        super(CustomModel, self).__init__(obs_space=obs_space, action_space=action_space, num_outputs=num_outputs, model_config=model_config, name=name)
        nn.Module.__init__(self)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(12)
        torch.cuda.manual_seed_all(12)
        self.critic_out = None

        self.matrix_net = nn.Sequential(
            nn.Conv2d(in_channels=obs_space.shape[1], out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
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
            nn.Flatten(),
            nn.Linear(800, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1, bias=False),
        )

        self.actor_net = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(800, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4, bias=False),
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens) -> TensorType:
        matrix_input: torch.Tensor = input_dict["obs"]
        if matrix_input.size(1) == 1:
            matrix_input = matrix_input.squeeze(dim=1)
        out = self.matrix_net(matrix_input)
        policy = self.actor_net(out)
        self.critic_out = self.critic_net(out)
        return policy, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return self.critic_out.view(-1)
