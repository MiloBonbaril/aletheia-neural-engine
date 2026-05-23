import os
import time
import random
import numpy as np
import torch
from collections import deque

from config import Config
from gym_env import GymnasiumEnv
from agent import PPOAgent

def evaluate_agent(agent: PPOAgent, env_id: str, n_episodes: int = 5) -> float:
    """
    Evaluates the agent's current policy deterministically.
    """
    eval_env = GymnasiumEnv(env_id)
    episode_rewards = []
    
    for _ in range(n_episodes):
        state, _ = eval_env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action, _, _ = agent.act(state, deterministic=True)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            total_reward += reward
            done = terminated or truncated
            
        episode_rewards.append(total_reward)
        
    eval_env.close()
    return float(np.mean(episode_rewards))

def main():
    # Load configuration parameters
    config = Config()
    
    # Seeding for deterministic behavior
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        
    print("=" * 60)
    print(f"Starting PPO Training for: {config.env_id}")
    print(f"Device: {config.device.upper()}")
    print(f"Max interaction steps: {config.total_timesteps:,}")
    print(f"Rollout horizon per update: {config.rollout_length}")
    print(f"Mini-batch size: {config.batch_size}")
    print("=" * 60)
    
    # Initialize unified Environment interface wrapper
    env = GymnasiumEnv(config.env_id)
    
    obs_dim = env.observation_shape[0]
    action_dim = env.action_shape[0]
    
    # Initialize PPO Agent
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config)
    
    # Track statistics
    recent_rewards = deque(maxlen=20)
    best_eval_reward = -float("inf")
    
    state, _ = env.reset(seed=config.seed)
    episode_reward = 0.0
    episode_length = 0
    num_episodes = 0
    
    start_time = time.time()
    global_step = 0
    update_count = 0
    
    while global_step < config.total_timesteps:
        # 1. Collect Rollout Trajectories
        for _ in range(config.rollout_length):
            action, log_prob, value = agent.act(state, deterministic=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            global_step += 1
            episode_reward += reward
            episode_length += 1
            
            # Mask is 0.0 if episode is terminated, 1.0 otherwise.
            # Truncation indicates environment limit, which is handled as non-terminal for bootstrapping.
            mask = 0.0 if terminated else 1.0
            
            # Store in rollout buffer
            agent.buffer.insert(state, action, log_prob, reward, value, mask)
            
            state = next_state
            
            if terminated or truncated:
                recent_rewards.append(episode_reward)
                num_episodes += 1
                
                # Check for evaluation interval
                if num_episodes % config.eval_interval_episodes == 0:
                    eval_mean = evaluate_agent(agent, config.env_id, config.eval_episodes)
                    print(f"\n---> Evaluation at step {global_step:,} over {config.eval_episodes} episodes: Mean Reward = {eval_mean:.2f}")
                    
                    if eval_mean > best_eval_reward:
                        best_eval_reward = eval_mean
                        agent.save(config.best_model_name)
                        print(f"[* NEW BEST CHECKPOINT SAVED *] Saved to {os.path.join(config.checkpoint_dir, config.best_model_name)} with score {eval_mean:.2f}")
                
                # Print episode statistics
                avg_recent = np.mean(recent_rewards) if recent_rewards else 0.0
                fps = int(global_step / (time.time() - start_time))
                print(f"Step: {global_step:7,} | Episode: {num_episodes:4} | Len: {episode_length:4} | Reward: {episode_reward:7.2f} | Avg(20): {avg_recent:7.2f} | FPS: {fps}")
                
                # Reset environment for new episode
                state, _ = env.reset()
                episode_reward = 0.0
                episode_length = 0
                
        # 2. Bootstrap Value Estimation of Final State
        obs_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            next_value = agent.critic(obs_t).item()
            
        next_mask = 0.0 if (terminated and not truncated) else 1.0
        
        # 3. Compute Advantages
        agent.buffer.compute_advantages(next_value, next_mask, config.gamma, config.gae_lambda)
        
        # 4. Perform Optimizer Parameter Updates
        metrics = agent.update()
        update_count += 1
        
        # 5. Decay Learning Rate if requested
        if config.lr_decay:
            agent.decay_lr(global_step, config.total_timesteps)
            
        # Log update metrics occasionally
        current_lr = agent.optimizer.param_groups[0]['lr']
        std = torch.exp(agent.actor.log_std).mean().item()
        print(f"--- UPDATE {update_count:3} --- Steps: {global_step:,} | LR: {current_lr:.2e} | Policy Loss: {metrics['loss_policy']:.4f} | Value Loss: {metrics['loss_value']:.4f} | Entropy: {metrics['entropy']:.4f} | Action Std: {std:.3f}")

    # Save final model
    agent.save(config.last_model_name)
    print("\n" + "=" * 60)
    print(f"Training Complete! Saved final model to: {os.path.join(config.checkpoint_dir, config.last_model_name)}")
    print(f"Best Evaluation Reward: {best_eval_reward:.2f}")
    print("=" * 60)
    
    env.close()

if __name__ == "__main__":
    main()
