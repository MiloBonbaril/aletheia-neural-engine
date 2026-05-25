import os
import argparse
import numpy as np
import torch
import time
import sys

# Allow imports from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config
from gym_env.gym_env import GymnasiumEnv
from agent.agent import PPOAgent

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent on continuous control environments.")
    parser.add_argument(
        "--checkpoint", 
        "-c",
        type=str, 
        default="checkpoints/ppo_bipedal_best.pt", 
        help="Path to the saved PyTorch model checkpoint."
    )
    parser.add_argument(
        "--episodes", 
        "-e",
        type=int, 
        default=5, 
        help="Number of evaluation episodes to run."
    )
    parser.add_argument(
        "--render", 
        "-r",
        action="store_true", 
        help="Render the environment visually during evaluation."
    )
    parser.add_argument(
        "--seed", 
        "-s",
        type=int, 
        default=100, 
        help="Random seed for evaluation environment initialization."
    )
    
    args = parser.parse_args()
    
    # 1. Load config and set up device
    config = Config()
    
    # If rendering is requested, set render_mode accordingly
    render_mode = "human" if args.render else None
    
    # 2. Re-create the Environment via the unified interface
    print(f"Creating evaluation environment: {config.env_id} with render_mode='{render_mode}'")
    try:
        env = GymnasiumEnv(config.env_id, render_mode=render_mode)
    except Exception as e:
        print(f"Error creating environment: {e}")
        if args.render:
            print("Note: Rendering may fail if running headlessly or if Pygame encounters display issues.")
            print("Retrying environment creation without rendering...")
            env = GymnasiumEnv(config.env_id, render_mode=None)
            args.render = False
        else:
            raise e
            
    obs_dim = env.observation_shape[0]
    action_dim = env.action_shape[0]
    
    # 3. Re-initialize Agent
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim, config=config)
    
    # 4. Load weights from checkpoint
    print(f"Loading model parameters from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file '{args.checkpoint}' not found!")
        print("Please train a model first using 'python train.py' to generate a checkpoint.")
        env.close()
        return
        
    agent.load(args.checkpoint)
    print("Model loaded successfully!")
    
    # 5. Run Evaluation Loop
    print("\n" + "=" * 60)
    print(f"Evaluating PPO Agent over {args.episodes} episodes...")
    print("=" * 60)
    
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            # Under evaluation, choose deterministic action (mean of distribution)
            action, _, _ = agent.act(state, deterministic=True)
            
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Simple frame capping to make human rendering visually regular
            if args.render:
                time.sleep(0.01)
                
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {ep:3d}: Reward = {total_reward:7.2f} | Steps = {steps:4d}")
        
    env.close()
    
    # 6. Display Summary Statistics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY:")
    print(f"Mean Episode Reward: {np.mean(episode_rewards):7.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Min Episode Reward : {np.min(episode_rewards):7.2f}")
    print(f"Max Episode Reward : {np.max(episode_rewards):7.2f}")
    print(f"Mean Episode Length: {np.mean(episode_lengths):7.1f} steps")
    print("=" * 60)

if __name__ == "__main__":
    main()
