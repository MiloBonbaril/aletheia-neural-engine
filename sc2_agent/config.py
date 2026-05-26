import os
import torch
from dataclasses import dataclass

@dataclass
class SC2Config:
    # Environment Settings
    map_name: str = "AcropolisLE"
    seed: int = 42
    
    # Feature Dimensions
    screen_resolution: int = 84
    minimap_resolution: int = 64
    screen_channels: int = 27       # Standard PySC2 screen layers
    minimap_channels: int = 11      # Standard PySC2 minimap layers
    
    # Entity Settings
    entity_features_dim: int = 30   # Attributes per unit (e.g. x, y, hp, type, alliance, etc.)
    entity_embed_dim: int = 128     # Transformer embedding size
    max_entities: int = 128         # Max units tracked per timestep (padded)
    
    # Scalar Settings
    scalar_features_dim: int = 11   # Economy and global metadata
    scalar_embed_dim: int = 64      # Dense output size for scalar features
    
    # Combined Spatial settings
    spatial_embed_dim: int = 128
    
    # Memory Core (Hybrid Mamba + Attention)
    d_model: int = 256              # Internal model dimension
    mamba_d_state: int = 16         # SSM state dimension
    mamba_d_conv: int = 4           # SSM convolution width
    mamba_expand: int = 2           # SSM expansion factor
    transformer_heads: int = 4      # Number of attention heads
    core_num_layers: int = 4        # Alternating layers count
    
    # Hierarchical RL (HRL) Settings
    macro_frequency: int = 8        # Step frequency for high-level macro decisions
    num_macro_goals: int = 10       # Total strategic macro goals
    macro_goal_embed_dim: int = 32  # Dense representation of the selected goal
    
    # Action Head Dimensions
    num_actions: int = 564          # Total actions in PySC2 action space
    spatial_resolution: int = 64    # Spatial coordination arg resolution (64x64 grid)
    
    # RL/PPO Hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    batch_size: int = 32
    rollout_length: int = 1024
    
    # Optimization coefficients
    lr_macro: float = 1e-4          # Learning rate for strategic Macro-Manager
    lr_micro: float = 3e-4          # Learning rate for tactical Micro-Manager
    c1_value_loss_coeff: float = 0.5
    c2_entropy_coeff: float = 0.01  # Exploration incentive
    max_grad_norm: float = 0.5
    
    # Directories & Device
    checkpoint_dir: str = "checkpoints_sc2"
    best_model_name: str = "sc2_hrl_best.pt"
    last_model_name: str = "sc2_hrl_last.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
