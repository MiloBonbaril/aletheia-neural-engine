import numpy as np
import torch
from typing import Generator, Tuple, Dict

class RolloutBuffer:
    """
    Rollout buffer to store trajectories collected from the environment during on-policy rolls.
    Computes Generalized Advantage Estimation (GAE) and returns target values.
    """
    def __init__(self, size: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...], device: str = "cpu"):
        self.size = size
        self.device = device
        
        # Experience storage
        self.obs = np.zeros((size,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((size,) + action_shape, dtype=np.float32)
        self.log_probs = np.zeros((size,), dtype=np.float32)
        self.rewards = np.zeros((size,), dtype=np.float32)
        self.values = np.zeros((size,), dtype=np.float32)
        self.masks = np.zeros((size,), dtype=np.float32)
        
        # Computed targets
        self.advantages = np.zeros((size,), dtype=np.float32)
        self.returns = np.zeros((size,), dtype=np.float32)
        
        self.ptr = 0

    def insert(self, obs: np.ndarray, action: np.ndarray, log_prob: float, reward: float, value: float, mask: float) -> None:
        """
        Inserts a single transitions step into the buffer.
        """
        assert self.ptr < self.size, "RolloutBuffer is already full! Clear it or call compute_advantages first."
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.masks[self.ptr] = mask
        self.ptr += 1

    def clear(self) -> None:
        """
        Resets the write pointer.
        """
        self.ptr = 0

    def compute_advantages(self, next_value: float, next_mask: float, gamma: float, gae_lambda: float) -> None:
        """
        Computes Generalized Advantage Estimations (GAE) and bootstrap returns.
        """
        last_gae_lam = 0.0
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_val = next_value
                next_non_terminal = next_mask
            else:
                next_val = self.values[t + 1]
                next_non_terminal = self.masks[t + 1]
            
            # Temporal Difference error
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            # Exponentially weighted GAE
            self.advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        
        # Target values for the Critic network (V_target = Advantage + V_estimate)
        self.returns = self.advantages + self.values

    def get_generator(self, batch_size: int) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yields randomized mini-batches of transitions for PPO network optimization.
        """
        assert self.ptr == self.size, f"Buffer is not fully populated (ptr={self.ptr}, size={self.size})!"
        
        indices = np.arange(self.size)
        np.random.shuffle(indices)
        
        # Normalize advantages across the entire rollout to stabilize training updates
        advantages_norm = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        
        for start in range(0, self.size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]
            
            # Pack mini-batch as PyTorch tensors moved to target hardware device
            yield {
                "obs": torch.tensor(self.obs[batch_idx], dtype=torch.float32, device=self.device),
                "actions": torch.tensor(self.actions[batch_idx], dtype=torch.float32, device=self.device),
                "log_probs": torch.tensor(self.log_probs[batch_idx], dtype=torch.float32, device=self.device),
                "values": torch.tensor(self.values[batch_idx], dtype=torch.float32, device=self.device),
                "advantages": torch.tensor(advantages_norm[batch_idx], dtype=torch.float32, device=self.device),
                "returns": torch.tensor(self.returns[batch_idx], dtype=torch.float32, device=self.device)
            }
