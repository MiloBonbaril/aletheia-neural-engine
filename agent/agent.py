import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any

from config import Config
from agent.networks import ActorNetwork, CriticNetwork
from agent.buffer import RolloutBuffer

class PPOAgent:
    """
    PPOAgent implementing Proximal Policy Optimization for continuous control tasks.
    Manages networks, rollout trajectory collections, value/policy updates, and saving/loading.
    """
    def __init__(self, obs_dim: int, action_dim: int, config: Config):
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(config.device)

        # Networks
        self.actor = ActorNetwork(
            obs_dim=obs_dim, 
            action_dim=action_dim, 
            hidden_dims=config.actor_hidden_dims,
            init_log_std=config.init_log_std
        ).to(self.device)
        
        self.critic = CriticNetwork(
            obs_dim=obs_dim, 
            hidden_dims=config.critic_hidden_dims
        ).to(self.device)

        # Combined optimizer with standard stabilization epsilon
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config.lr,
            eps=1e-5
        )

        # Rollout storage buffer
        self.buffer = RolloutBuffer(
            size=config.rollout_length,
            obs_shape=(obs_dim,),
            action_shape=(action_dim,),
            device=config.device
        )

        # Pré-allocation d'un tenseur d'observation pour éviter les instanciations à la volée
        # Trick de sioux : on réutilise la mémoire !
        self._obs_buffer = torch.zeros((1, obs_dim), dtype=torch.float32, device=self.device)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Samples an action based on current state observations.

        Args:
            obs (np.ndarray): State observation from the environment.
            deterministic (bool): If True, returns action mean without exploration noise.

        Returns:
            action (np.ndarray): The chosen continuous action values.
            log_prob (float): Log probability density of the chosen action (0.0 if deterministic).
            value (float): Critic state-value estimation (0.0 if deterministic).
        """
        # Convert state observation to tensor
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            self._obs_buffer[0].copy_(torch.from_numpy(obs))
            dist, action_mean = self.actor(self._obs_buffer)
            
            if deterministic:
                return action_mean.squeeze(0).cpu().numpy(), 0.0, 0.0
            
            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t).sum(dim=-1)
            value_t = self.critic(self._obs_buffer).squeeze(0)
            
        return (
            action_t.squeeze(0).cpu().numpy(),
            log_prob_t.item(),
            value_t.item()
        )

    def decay_lr(self, current_step: int, total_steps: int, start_step: int = 0, start_lr: float = None) -> None:
        """
        Linearly decays learning rate based on current progress from start_step to total_steps.
        """
        if start_lr is None:
            start_lr = self.config.lr
            
        denom = total_steps - start_step
        if denom <= 0:
            fraction = 0.0
        else:
            fraction = (total_steps - current_step) / denom
            
        new_lr = start_lr * max(fraction, 0.0)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def update(self) -> Dict[str, float]:
        """
        Runs optimization updates across all gathered experiences in the buffer.
        """
        policy_loss_epoch = 0.0
        value_loss_epoch = 0.0
        entropy_loss_epoch = 0.0
        kl_divergence_epoch = 0.0
        num_batches = 0
        
        early_stopped = False

        for epoch in range(self.config.ppo_epochs):
            if early_stopped:
                break
                
            mini_batch_gen = self.buffer.get_generator(self.config.batch_size)
            
            for batch in mini_batch_gen:
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # --- TRICK 1 : Normalisation des avantages au niveau du mini-batch ---
                # Cela stabilise le gradient de l'acteur de façon exponentielle !
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Forward pass
                dist, _ = self.actor(obs)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                new_values = self.critic(obs).squeeze(-1)

                # Policy ratio: r_t(theta)
                ratios = torch.exp(new_log_probs - old_log_probs)

                # Calcul de la KL approximative AVANT la mise à jour pour le monitoring
                with torch.no_grad():
                    log_ratio = new_log_probs - old_log_probs
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                
                # --- TRICK 2 : Early Stopping si la politique dérive trop ---
                # Si la configuration spécifie un target_kl (ex: 0.015), on coupe le massacre.
                if hasattr(self.config, 'target_kl') and approx_kl > self.config.target_kl:
                    early_stopped = True
                    break

                # PPO Clip Loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic value loss using standard MSE
                value_loss = 0.5 * nn.MSELoss()(new_values, returns)

                # Combined scalar loss
                total_loss = (
                    policy_loss + 
                    self.config.c1_value_loss_coeff * value_loss - 
                    self.config.c2_entropy_coeff * entropy
                )

                # Backprop nerveuse
                self.optimizer.zero_grad(set_to_none=True) # set_to_none=True évite d'écrire des zéros, économise de la bande passante mémoire !
                total_loss.backward()
                
                # Clipping des gradients pour éviter les explosions numériques
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()

                # Accumulation des métriques
                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_loss_epoch += entropy.item()
                kl_divergence_epoch += approx_kl
                num_batches += 1

        # Clear rollout experiences buffer for the next collection phase
        self.buffer.clear()

        # Protection contre la division par zéro si early stop au premier batch
        safe_num_batches = max(num_batches, 1)

        return {
            "loss_policy": policy_loss_epoch / safe_num_batches,
            "loss_value": value_loss_epoch / safe_num_batches,
            "entropy": entropy_loss_epoch / safe_num_batches,
            "kl_divergence": kl_divergence_epoch / safe_num_batches
        }

    def save(
        self, 
        filename: str, 
        global_step: int = 0, 
        num_episodes: int = 0, 
        best_eval_reward: float = -float("inf"),
        update_count: int = 0,
        recent_rewards: list = None
    ) -> None:
        """
        Saves policy parameters and optimization checkpoints.
        """
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "global_step": global_step,
            "num_episodes": num_episodes,
            "best_eval_reward": best_eval_reward,
            "update_count": update_count,
            "recent_rewards": recent_rewards if recent_rewards is not None else []
        }, filepath)

    def load(self, filepath: str) -> Dict[str, Any]:
        """
        Loads policy parameters and optimization checkpoints.
        Returns a dictionary containing training metadata.
        """
        assert os.path.exists(filepath), f"Checkpoint not found at: {filepath}"
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        return {
            "global_step": checkpoint.get("global_step", 0),
            "num_episodes": checkpoint.get("num_episodes", 0),
            "best_eval_reward": checkpoint.get("best_eval_reward", -float("inf")),
            "update_count": checkpoint.get("update_count", 0),
            "recent_rewards": checkpoint.get("recent_rewards", []),
            "config": checkpoint.get("config", self.config)
        }
