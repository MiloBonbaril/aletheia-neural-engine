import os
import torch
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class Config:
    # Environment Settings
    env_id: str = "BipedalWalker-v3"
    seed: int = 42
    
    # RL Agent (PPO) Hyperparameters
    gamma: float = 0.99                # Discount factor
    gae_lambda: float = 0.95           # GAE parameter for reward shaping
    clip_epsilon: float = 0.2          # PPO policy clipping parameter
    ppo_epochs: int = 10               # Number of optimization epochs per rollout
    batch_size: int = 64               # Mini-batch size
    rollout_length: int = 2048         # Number of steps collected per rollout
    target_kl: float = 0.015           # Target KL divergence for early stopping
    
    # Learning Rates and Regularization
    lr: float = 3e-4                   # Learning rate for Adam
    lr_decay: bool = True              # Linearly decay learning rate over training
    max_grad_norm: float = 0.5         # Maximum gradient clipping norm
    c1_value_loss_coeff: float = 0.5   # Value loss coefficient
    c2_entropy_coeff: float = 0.01     # Entropy coefficient to encourage exploration
    
    # Network Architecture
    actor_hidden_dims: Tuple[int, ...] = (256, 256)
    critic_hidden_dims: Tuple[int, ...] = (256, 256)
    
    # Action exploration noise
    init_log_std: float = -0.5         # Initial log standard deviation for actions
    
    # Training Loop Configurations
    total_timesteps: int = 1_000_000   # Total environment interaction steps
    eval_interval_episodes: int = 50   # How often to evaluate the policy
    eval_episodes: int = 5             # Number of episodes for evaluation
    
    # Checkpointing and Directories
    checkpoint_dir: str = "checkpoints"
    best_model_name: str = "opt_ppo_bipedal_best.pt"
    last_model_name: str = "opt_ppo_bipedal_last.pt"
    
    # Computation Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
