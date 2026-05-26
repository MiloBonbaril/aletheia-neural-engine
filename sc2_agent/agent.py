import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, Optional

from sc2_agent.config import SC2Config
from sc2_agent.model import SC2HierarchicalModel
from sc2_agent.buffer import SC2HierarchicalRolloutBuffer

class SC2HierarchicalAgent:
    """
    SC2HierarchicalAgent implementing Hierarchical PPO (HRL) for StarCraft II.
    Manages dual Macro/Micro networks, dynamic available actions masking, 
    recurrent SSM-Attention states, rollout trajectory collection, and updates on CUDA.
    """
    def __init__(self, config: SC2Config):
        self.config = config
        self.device = torch.device(config.device)
        
        # Unified Model
        self.model = SC2HierarchicalModel(config).to(self.device)
        
        # Dual learning rate optimizers
        self.macro_optimizer = optim.Adam(
            list(self.model.macro_policy.parameters()) + list(self.model.macro_critic.parameters()),
            lr=config.lr_macro,
            eps=1e-5
        )
        self.micro_optimizer = optim.Adam(
            list(self.model.spatial_torso.parameters()) + 
            list(self.model.entity_torso.parameters()) + 
            list(self.model.scalar_torso.parameters()) +
            list(self.model.core_proj.parameters()) +
            list(self.model.core.parameters()) +
            list(self.model.micro_policy.parameters()) + 
            list(self.model.micro_critic.parameters()),
            lr=config.lr_micro,
            eps=1e-5
        )
        
        # Rollout Buffer
        self.buffer = SC2HierarchicalRolloutBuffer(config.rollout_length, config)
        
        # --- Recurrent Memory Context ---
        self.d_inner = config.d_model * config.mamba_expand
        self.ssm_state = torch.zeros((1, self.d_inner, config.mamba_d_state), dtype=torch.float32, device=self.device)
        self.attn_history = None
        self.history_limit = 64 # Max history length for causal self-attention
        
        # Active High-Level strategic goal tracking
        self.current_macro_goal = None
        self.current_macro_log_prob = 0.0
        self.step_counter = 0

    def reset_memory(self) -> None:
        """
        Resets recurrent state space and strategic goals at episode boundaries.
        """
        self.ssm_state.zero_()
        self.attn_history = None
        self.current_macro_goal = None
        self.current_macro_log_prob = 0.0
        self.step_counter = 0

    def act(
        self,
        screen: np.ndarray,            # (C_s, 84, 84)
        minimap: np.ndarray,           # (C_m, 64, 64)
        entities: np.ndarray,          # (N, F_e)
        entity_mask: np.ndarray,       # (N,) bool
        scalars: np.ndarray,           # (F_s,)
        avail_actions: np.ndarray      # (num_actions,) bool
    ) -> Tuple[Dict[str, int], Dict[str, float], Tuple[float, float], np.ndarray]:
        """
        Computes hierarchical actor decisions for a single step.
        """
        # Convert inputs to PyTorch tensors with explicit Batch and Temporal sequence dimensions -> (B=1, L=1, ...)
        screen_t = torch.tensor(screen, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        minimap_t = torch.tensor(minimap, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        entities_t = torch.tensor(entities, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        entity_mask_t = torch.tensor(entity_mask, dtype=torch.bool, device=self.device).unsqueeze(0).unsqueeze(0)
        scalars_t = torch.tensor(scalars, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        avail_mask_t = torch.tensor(avail_actions, dtype=torch.bool, device=self.device).unsqueeze(0).unsqueeze(0)
        
        # Capture current SSM state before forward pass (to store in buffer)
        saved_ssm_state = self.ssm_state.squeeze(0).cpu().numpy()
        
        with torch.no_grad():
            # 1. Evaluate/sample Macro Goal if frequency matches or goal is unassigned
            if self.current_macro_goal is None or self.step_counter % self.config.macro_frequency == 0:
                actions, v_macro, v_micro, next_ssm, next_attn, macro_logits = self.model(
                    screen=screen_t,
                    minimap=minimap_t,
                    entities=entities_t,
                    entity_padding_mask=entity_mask_t,
                    scalars=scalars_t,
                    available_actions_mask=avail_mask_t,
                    ssm_state=self.ssm_state,
                    attn_history=self.attn_history,
                    macro_goal=None
                )
                self.current_macro_goal = actions["macro_goal"]
                self.current_macro_log_prob = actions["log_prob_macro"].item()
            else:
                # Reuse strategic goal, compute actions based on active goal
                actions, v_macro, v_micro, next_ssm, next_attn, macro_logits = self.model(
                    screen=screen_t,
                    minimap=minimap_t,
                    entities=entities_t,
                    entity_padding_mask=entity_mask_t,
                    scalars=scalars_t,
                    available_actions_mask=avail_mask_t,
                    ssm_state=self.ssm_state,
                    attn_history=self.attn_history,
                    macro_goal=self.current_macro_goal
                )
            
            # Step recurrent states forward
            self.ssm_state = next_ssm
            
            # Maintain sliding attention history cache
            if next_attn is not None:
                if self.attn_history is None:
                    self.attn_history = next_attn
                else:
                    # Append and crop
                    combined = torch.cat([self.attn_history, next_attn], dim=1)
                    if combined.shape[1] > self.history_limit:
                        combined = combined[:, -self.history_limit:, :]
                    self.attn_history = combined.detach()
            
            self.step_counter += 1
            
        # Format returns
        decision_actions = {
            "macro_goal": self.current_macro_goal.item(),
            "a_major": actions["a_major"].item(),
            "a_x": actions["a_x"].item(),
            "a_y": actions["a_y"].item(),
            "a_unit": actions["a_unit"].item()
        }
        
        log_probs = {
            "macro": self.current_macro_log_prob,
            "major": actions["log_prob_major"].item(),
            "x": actions["log_prob_x"].item(),
            "y": actions["log_prob_y"].item(),
            "unit": actions["log_prob_unit"].item()
        }
        
        values = (v_macro.item(), v_micro.item())
        
        return decision_actions, log_probs, values, saved_ssm_state

    def update(self) -> Dict[str, float]:
        """
        Performs Hierarchical recurrent PPO optimization updates.
        """
        metrics = {
            "macro_loss_policy": 0.0, "macro_loss_value": 0.0,
            "micro_loss_policy": 0.0, "micro_loss_value": 0.0,
            "num_batches": 0
        }
        
        # Sequential updates over PPO epochs
        for epoch in range(self.config.ppo_epochs):
            # Specialized generator yielding sequence batches
            mini_batch_gen = self.buffer.get_recurrent_generator(self.config.batch_size)
            
            for batch in mini_batch_gen:
                screen = batch["screen"]                       # (B, L, C_s, 84, 84)
                minimap = batch["minimap"]                     # (B, L, C_m, 64, 64)
                entities = batch["entities"]                   # (B, L, N, F_e)
                ent_mask = batch["entity_padding_mask"]       # (B, L, N)
                scalars = batch["scalars"]                     # (B, L, F_s)
                avail_mask = batch["available_actions_mask"]   # (B, L, num_actions)
                init_ssm = batch["init_ssm_state"]             # (B, D_inner, D_state)
                
                macro_goals = batch["macro_goal"]             # (B, L)
                a_major = batch["a_major"]                     # (B, L)
                a_x = batch["a_x"]                             # (B, L)
                a_y = batch["a_y"]                             # (B, L)
                a_unit = batch["a_unit"]                       # (B, L)
                
                old_log_probs_macro = batch["log_prob_macro"]   # (B, L)
                old_log_probs_major = batch["log_prob_major"]   # (B, L)
                old_log_probs_x = batch["log_prob_x"]           # (B, L)
                old_log_probs_y = batch["log_prob_y"]           # (B, L)
                old_log_probs_unit = batch["log_prob_unit"]     # (B, L)
                
                returns_macro = batch["returns_macro"]         # (B, L)
                returns_micro = batch["returns_micro"]         # (B, L)
                adv_macro = batch["advantages_macro"]           # (B, L)
                adv_micro = batch["advantages_micro"]           # (B, L)
                
                # Retrieve actions taken to evaluate exact joint log-probs during forward pass
                actions_taken = {
                    "major": a_major,
                    "x": a_x,
                    "y": a_y,
                    "unit": a_unit
                }
                
                # --- MODEL FORWARD PASS ---
                # We feed the sequential chunk batch through the model with causal masking
                actions_pred, v_macro, v_micro, _, _, macro_logits = self.model(
                    screen=screen,
                    minimap=minimap,
                    entities=entities,
                    entity_padding_mask=ent_mask,
                    scalars=scalars,
                    available_actions_mask=avail_mask,
                    ssm_state=init_ssm,
                    attn_history=None, # In sequence training, causal masking computes all tokens in 1 pass
                    macro_goal=macro_goals,
                    actions_taken=actions_taken
                )
                
                # ----------------------------------------------------
                # 1. Macro-Manager PPO Updates
                # ----------------------------------------------------
                new_log_macro = actions_pred["log_prob_macro"]
                ratio_macro = torch.exp(new_log_macro - old_log_probs_macro)
                
                surr1_macro = ratio_macro * adv_macro
                surr2_macro = torch.clamp(ratio_macro, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * adv_macro
                macro_policy_loss = -torch.min(surr1_macro, surr2_macro).mean()
                
                macro_val_loss = 0.5 * nn.MSELoss()(v_macro, returns_macro)
                
                # Combined Macro Loss
                macro_entropy = -(F.softmax(macro_logits, dim=-1) * F.log_softmax(macro_logits, dim=-1)).sum(dim=-1).mean()
                total_loss_macro = (
                    macro_policy_loss + 
                    self.config.c1_value_loss_coeff * macro_val_loss - 
                    self.config.c2_entropy_coeff * macro_entropy
                )
                
                # ----------------------------------------------------
                # 2. Micro-Manager PPO Updates (Combined Joint Action Log-Probs)
                # ----------------------------------------------------
                new_log_major = actions_pred["log_prob_major"]
                new_log_x = actions_pred["log_prob_x"]
                new_log_y = actions_pred["log_prob_y"]
                new_log_unit = actions_pred["log_prob_unit"]
                
                # Compute joint probabilities
                new_log_micro_joint = new_log_major + new_log_x + new_log_y + new_log_unit
                old_log_micro_joint = old_log_probs_major + old_log_probs_x + old_log_probs_y + old_log_probs_unit
                
                ratio_micro = torch.exp(new_log_micro_joint - old_log_micro_joint)
                
                surr1_micro = ratio_micro * adv_micro
                surr2_micro = torch.clamp(ratio_micro, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * adv_micro
                micro_policy_loss = -torch.min(surr1_micro, surr2_micro).mean()
                
                micro_val_loss = 0.5 * nn.MSELoss()(v_micro, returns_micro)
                
                # Combined Micro Loss
                total_loss_micro = (
                    micro_policy_loss + 
                    self.config.c1_value_loss_coeff * micro_val_loss
                )
                
                # --- BACKPROPAGATION ---
                # Backprop Macro branch
                self.macro_optimizer.zero_grad(set_to_none=True)
                total_loss_macro.backward(retain_graph=True) # Share torsos gradients properly if any overlap
                nn.utils.clip_grad_norm_(
                    list(self.model.macro_policy.parameters()) + list(self.model.macro_critic.parameters()), 
                    self.config.max_grad_norm
                )
                self.macro_optimizer.step()
                
                # Backprop Micro branch
                self.micro_optimizer.zero_grad(set_to_none=True)
                total_loss_micro.backward()
                nn.utils.clip_grad_norm_(
                    list(self.model.spatial_torso.parameters()) + 
                    list(self.model.entity_torso.parameters()) + 
                    list(self.model.scalar_torso.parameters()) +
                    list(self.model.core_proj.parameters()) +
                    list(self.model.core.parameters()) +
                    list(self.model.micro_policy.parameters()) + 
                    list(self.model.micro_critic.parameters()), 
                    self.config.max_grad_norm
                )
                self.micro_optimizer.step()
                
                # Log metrics
                metrics["macro_loss_policy"] += macro_policy_loss.item()
                metrics["macro_loss_value"] += macro_val_loss.item()
                metrics["micro_loss_policy"] += micro_policy_loss.item()
                metrics["micro_loss_value"] += micro_val_loss.item()
                metrics["num_batches"] += 1
                
        # Clear buffer
        self.buffer.clear()
        
        # Normalize metrics
        safe_batches = max(metrics["num_batches"], 1)
        return {k: v / safe_batches for k, v in metrics.items() if k != "num_batches"}

    def save(self, filename: str, **kwargs) -> None:
        """
        Saves full Hierarchical models, optimizers, and checkpoint statistics.
        """
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "macro_optimizer_state_dict": self.macro_optimizer.state_dict(),
            "micro_optimizer_state_dict": self.micro_optimizer.state_dict(),
            "config": self.config
        }
        save_dict.update(kwargs)
        torch.save(save_dict, filepath)

    def load(self, filepath: str) -> Dict[str, Any]:
        """
        Loads hierarchical model state dicts from checkpoints.
        """
        assert os.path.exists(filepath), f"Checkpoint '{filepath}' not found!"
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "macro_optimizer_state_dict" in checkpoint:
            self.macro_optimizer.load_state_dict(checkpoint["macro_optimizer_state_dict"])
        if "micro_optimizer_state_dict" in checkpoint:
            self.micro_optimizer.load_state_dict(checkpoint["micro_optimizer_state_dict"])
        return checkpoint
