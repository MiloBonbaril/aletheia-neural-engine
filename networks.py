import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, List

def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """
    Applies orthogonal initialization to a linear layer.
    Orthogonal initialization is critical for training stability in continuous control tasks.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorNetwork(nn.Module):
    """
    Actor network mapping state observations to a Gaussian distribution over continuous actions.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (256, 256), init_log_std: float = -0.5):
        super().__init__()
        
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            layers.append(nn.Tanh())
            in_dim = h_dim
        
        # The mean network (outputs actions scaled to [-1, 1] using Tanh)
        self.mean_net = nn.Sequential(*layers)
        self.mean_layer = layer_init(nn.Linear(in_dim, action_dim), std=0.01)
        
        # Action standard deviation is modeled as state-independent learnable parameters
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """
        Returns a PyTorch Normal distribution over the actions and the mean actions.
        """
        hidden = self.mean_net(x)
        action_mean = torch.tanh(self.mean_layer(hidden))
        
        # Convert log standard deviation to standard deviation
        action_std = torch.exp(self.log_std)
        # Create a batched diagonal normal distribution
        dist = Normal(action_mean, action_std)
        
        return dist, action_mean


class CriticNetwork(nn.Module):
    """
    Critic network mapping state observations to state-value estimates V(s).
    """
    def __init__(self, obs_dim: int, hidden_dims: Tuple[int, ...] = (256, 256)):
        super().__init__()
        
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for h_dim in hidden_dims:
            layers.append(layer_init(nn.Linear(in_dim, h_dim)))
            layers.append(nn.Tanh())
            in_dim = h_dim
            
        self.val_net = nn.Sequential(
            *layers,
            layer_init(nn.Linear(in_dim, 1), std=1.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimates the state value V(s).
        """
        return self.val_net(x)
