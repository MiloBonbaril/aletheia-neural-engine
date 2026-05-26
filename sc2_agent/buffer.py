import numpy as np
import torch
from typing import Generator, Dict, Any, Tuple

class SC2HierarchicalRolloutBuffer:
    """
    Specialized rollout buffer for Hierarchical and Recurrent RL.
    Stores multi-modal visual, entity, scalar observations, action choices, 
    advantages, and SSM cell states.
    Provides sequence mini-batch generators for recurrent PPO optimization.
    """
    def __init__(self, size: int, config: Any):
        self.size = size
        self.config = config
        self.device = config.device
        
        # --- State Observations Storage ---
        self.screen = np.zeros((size, config.screen_channels, config.screen_resolution, config.screen_resolution), dtype=np.float32)
        self.minimap = np.zeros((size, config.minimap_channels, config.minimap_resolution, config.minimap_resolution), dtype=np.float32)
        self.entities = np.zeros((size, config.max_entities, config.entity_features_dim), dtype=np.float32)
        self.entity_padding_mask = np.zeros((size, config.max_entities), dtype=bool)
        self.scalars = np.zeros((size, config.scalar_features_dim), dtype=np.float32)
        self.available_actions_mask = np.zeros((size, config.num_actions), dtype=bool)
        
        # --- Recurrent SSM States (Saved at each step to allow chunking) ---
        d_inner = config.d_model * config.mamba_expand
        self.ssm_states = np.zeros((size, d_inner, config.mamba_d_state), dtype=np.float32)
        
        # --- Action Decisions Storage ---
        self.macro_goals = np.zeros((size,), dtype=np.int64)
        self.a_major = np.zeros((size,), dtype=np.int64)
        self.a_x = np.zeros((size,), dtype=np.int64)
        self.a_y = np.zeros((size,), dtype=np.int64)
        self.a_unit = np.zeros((size,), dtype=np.int64)
        
        # --- Action Log-Probabilities ---
        self.log_probs_macro = np.zeros((size,), dtype=np.float32)
        self.log_probs_major = np.zeros((size,), dtype=np.float32)
        self.log_probs_x = np.zeros((size,), dtype=np.float32)
        self.log_probs_y = np.zeros((size,), dtype=np.float32)
        self.log_probs_unit = np.zeros((size,), dtype=np.float32)
        
        # --- Rewards & Masks ---
        self.rewards_macro = np.zeros((size,), dtype=np.float32)
        self.rewards_micro = np.zeros((size,), dtype=np.float32)
        self.masks = np.zeros((size,), dtype=np.float32) # 0.0 if terminal step, 1.0 otherwise
        
        # --- Critic Estimations ---
        self.values_macro = np.zeros((size,), dtype=np.float32)
        self.values_micro = np.zeros((size,), dtype=np.float32)
        
        # --- Advantages and Bootstrapped Target Returns ---
        self.advantages_macro = np.zeros((size,), dtype=np.float32)
        self.returns_macro = np.zeros((size,), dtype=np.float32)
        
        self.advantages_micro = np.zeros((size,), dtype=np.float32)
        self.returns_micro = np.zeros((size,), dtype=np.float32)
        
        self.ptr = 0

    def insert(
        self,
        screen: np.ndarray,
        minimap: np.ndarray,
        entities: np.ndarray,
        entity_mask: np.ndarray,
        scalars: np.ndarray,
        avail_actions: np.ndarray,
        ssm_state: np.ndarray,
        actions: Dict[str, int],
        log_probs: Dict[str, float],
        r_macro: float,
        r_micro: float,
        v_macro: float,
        v_micro: float,
        mask: float
    ) -> None:
        """
        Inserts a single transition step into the buffer.
        """
        assert self.ptr < self.size, "SC2HierarchicalRolloutBuffer is already full!"
        
        self.screen[self.ptr] = screen
        self.minimap[self.ptr] = minimap
        self.entities[self.ptr] = entities
        self.entity_padding_mask[self.ptr] = entity_mask
        self.scalars[self.ptr] = scalars
        self.available_actions_mask[self.ptr] = avail_actions
        self.ssm_states[self.ptr] = ssm_state
        
        self.macro_goals[self.ptr] = actions["macro_goal"]
        self.a_major[self.ptr] = actions["a_major"]
        self.a_x[self.ptr] = actions["a_x"]
        self.a_y[self.ptr] = actions["a_y"]
        self.a_unit[self.ptr] = actions["a_unit"]
        
        self.log_probs_macro[self.ptr] = log_probs["macro"]
        self.log_probs_major[self.ptr] = log_probs["major"]
        self.log_probs_x[self.ptr] = log_probs["x"]
        self.log_probs_y[self.ptr] = log_probs["y"]
        self.log_probs_unit[self.ptr] = log_probs["unit"]
        
        self.rewards_macro[self.ptr] = r_macro
        self.rewards_micro[self.ptr] = r_micro
        self.values_macro[self.ptr] = v_macro
        self.values_micro[self.ptr] = v_micro
        self.masks[self.ptr] = mask
        
        self.ptr += 1

    def clear(self) -> None:
        self.ptr = 0

    def compute_advantages(
        self,
        next_val_macro: float,
        next_val_micro: float,
        next_mask: float,
        gamma: float,
        gae_lambda: float
    ) -> None:
        """
        Computes Generalized Advantage Estimation (GAE) for both Macro and Micro agents.
        """
        last_gae_macro = 0.0
        last_gae_micro = 0.0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_v_macro = next_val_macro
                next_v_micro = next_val_micro
                next_non_terminal = next_mask
            else:
                next_v_macro = self.values_macro[t + 1]
                next_v_micro = self.values_micro[t + 1]
                next_non_terminal = self.masks[t + 1]
            
            # 1. Macro Advantages
            delta_macro = self.rewards_macro[t] + gamma * next_v_macro * next_non_terminal - self.values_macro[t]
            self.advantages_macro[t] = last_gae_macro = delta_macro + gamma * gae_lambda * next_non_terminal * last_gae_macro
            
            # 2. Micro Advantages
            delta_micro = self.rewards_micro[t] + gamma * next_v_micro * next_non_terminal - self.values_micro[t]
            self.advantages_micro[t] = last_gae_micro = delta_micro + gamma * gae_lambda * next_non_terminal * last_gae_micro
            
        # Target values for critics
        self.returns_macro = self.advantages_macro + self.values_macro
        self.returns_micro = self.advantages_micro + self.values_micro

    def get_recurrent_generator(self, batch_size: int, chunk_len: int = 32) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Splits rollout into randomized sequential chunks for stable Recurrent PPO training.
        """
        assert self.ptr == self.size, "Buffer is not fully populated!"
        
        # Calculate number of sequential chunks in the buffer
        n_chunks = self.size // chunk_len
        indices = np.arange(n_chunks)
        np.random.shuffle(indices)
        
        # Normalize advantages across the entire buffer to stabilize gradient steps
        adv_macro_norm = (self.advantages_macro - self.advantages_macro.mean()) / (self.advantages_macro.std() + 1e-8)
        adv_micro_norm = (self.advantages_micro - self.advantages_micro.mean()) / (self.advantages_micro.std() + 1e-8)
        
        # Determine how many chunks fit in a single batch
        chunks_per_batch = batch_size // chunk_len
        
        for start in range(0, n_chunks, chunks_per_batch):
            batch_chunk_indices = indices[start:start+chunks_per_batch]
            
            # Pack chunk sequences
            seq_screens, seq_minimaps, seq_entities, seq_entity_masks = [], [], [], []
            seq_scalars, seq_avail_actions, init_ssm_states = [], [], []
            
            seq_macro_goals, seq_a_major, seq_a_x, seq_a_y, seq_a_unit = [], [], [], [], []
            seq_log_macro, seq_log_major, seq_log_x, seq_log_y, seq_log_unit = [], [], [], [], []
            seq_ret_macro, seq_ret_micro, seq_adv_macro, seq_adv_micro = [], [], [], []
            
            for chunk_idx in batch_chunk_indices:
                start_step = chunk_idx * chunk_len
                end_step = start_step + chunk_len
                
                # Retrieve initial SSM state for the beginning of this sequential chunk
                init_ssm_states.append(self.ssm_states[start_step])
                
                # Append slice sequences
                seq_screens.append(self.screen[start_step:end_step])
                seq_minimaps.append(self.minimap[start_step:end_step])
                seq_entities.append(self.entities[start_step:end_step])
                seq_entity_masks.append(self.entity_padding_mask[start_step:end_step])
                seq_scalars.append(self.scalars[start_step:end_step])
                seq_avail_actions.append(self.available_actions_mask[start_step:end_step])
                
                seq_macro_goals.append(self.macro_goals[start_step:end_step])
                seq_a_major.append(self.a_major[start_step:end_step])
                seq_a_x.append(self.a_x[start_step:end_step])
                seq_a_y.append(self.a_y[start_step:end_step])
                seq_a_unit.append(self.a_unit[start_step:end_step])
                
                seq_log_macro.append(self.log_probs_macro[start_step:end_step])
                seq_log_major.append(self.log_probs_major[start_step:end_step])
                seq_log_x.append(self.log_probs_x[start_step:end_step])
                seq_log_y.append(self.log_probs_y[start_step:end_step])
                seq_log_unit.append(self.log_probs_unit[start_step:end_step])
                
                seq_ret_macro.append(self.returns_macro[start_step:end_step])
                seq_ret_micro.append(self.returns_micro[start_step:end_step])
                seq_adv_macro.append(adv_macro_norm[start_step:end_step])
                seq_adv_micro.append(adv_micro_norm[start_step:end_step])
                
            # Convert list arrays to tensors and move to GPU/CPU device
            yield {
                "screen": torch.tensor(np.array(seq_screens), dtype=torch.float32, device=self.device),
                "minimap": torch.tensor(np.array(seq_minimaps), dtype=torch.float32, device=self.device),
                "entities": torch.tensor(np.array(seq_entities), dtype=torch.float32, device=self.device),
                "entity_padding_mask": torch.tensor(np.array(seq_entity_masks), dtype=torch.bool, device=self.device),
                "scalars": torch.tensor(np.array(seq_scalars), dtype=torch.float32, device=self.device),
                "available_actions_mask": torch.tensor(np.array(seq_avail_actions), dtype=torch.bool, device=self.device),
                
                # Initial SSM state for the chunks shape: (B_chunk, D_inner, D_state)
                "init_ssm_state": torch.tensor(np.array(init_ssm_states), dtype=torch.float32, device=self.device),
                
                "macro_goal": torch.tensor(np.array(seq_macro_goals), dtype=torch.int64, device=self.device),
                "a_major": torch.tensor(np.array(seq_a_major), dtype=torch.int64, device=self.device),
                "a_x": torch.tensor(np.array(seq_a_x), dtype=torch.int64, device=self.device),
                "a_y": torch.tensor(np.array(seq_a_y), dtype=torch.int64, device=self.device),
                "a_unit": torch.tensor(np.array(seq_a_unit), dtype=torch.int64, device=self.device),
                
                "log_prob_macro": torch.tensor(np.array(seq_log_macro), dtype=torch.float32, device=self.device),
                "log_prob_major": torch.tensor(np.array(seq_log_major), dtype=torch.float32, device=self.device),
                "log_prob_x": torch.tensor(np.array(seq_log_x), dtype=torch.float32, device=self.device),
                "log_prob_y": torch.tensor(np.array(seq_log_y), dtype=torch.float32, device=self.device),
                "log_prob_unit": torch.tensor(np.array(seq_log_unit), dtype=torch.float32, device=self.device),
                
                "returns_macro": torch.tensor(np.array(seq_ret_macro), dtype=torch.float32, device=self.device),
                "returns_micro": torch.tensor(np.array(seq_ret_micro), dtype=torch.float32, device=self.device),
                "advantages_macro": torch.tensor(np.array(seq_adv_macro), dtype=torch.float32, device=self.device),
                "advantages_micro": torch.tensor(np.array(seq_adv_micro), dtype=torch.float32, device=self.device)
            }
