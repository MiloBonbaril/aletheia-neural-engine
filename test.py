import sys
import os
import torch
import numpy as np

# Append workspace directory so python can find the newly created sc2_agent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
# Also append workspace directory absolute path directly
sys.path.append("t:\\taf\\code\\aletheia-neural-engine")

from sc2_agent.config import SC2Config
from sc2_agent.agent import SC2HierarchicalAgent

def main():
    print("=" * 60)
    print("VALIDATING STARCRAFT II HIERARCHICAL AGENT & ARCHITECTURE")
    print("=" * 60)
    
    # 1. Initialize configuration and agent
    config = SC2Config()
    print(f"Target Compute Device: {config.device.upper()}")
    
    agent = SC2HierarchicalAgent(config)
    print("Agent initialized successfully!")
    
    # 2. Generate mock observation matching PySC2 specs
    np.random.seed(42)
    torch.manual_seed(42)
    
    screen = np.random.randn(config.screen_channels, config.screen_resolution, config.screen_resolution).astype(np.float32)
    minimap = np.random.randn(config.minimap_channels, config.minimap_resolution, config.minimap_resolution).astype(np.float32)
    
    # Entity specs: 128 units, each having 30 features
    entities = np.random.randn(config.max_entities, config.entity_features_dim).astype(np.float32)
    
    # Padding mask: let's pad the last 40 entities (mark as True/ignored)
    entity_mask = np.zeros((config.max_entities,), dtype=bool)
    entity_mask[-40:] = True
    
    scalars = np.random.randn(config.scalar_features_dim).astype(np.float32)
    
    # Legal actions: allow only a tiny set of actions, say indices [0, 5, 12, 100, 350]
    avail_actions = np.zeros((config.num_actions,), dtype=bool)
    legal_action_indices = [0, 5, 12, 100, 350]
    avail_actions[legal_action_indices] = True
    
    print("\n--- Running Act Step (Evaluation / Decision) ---")
    actions, log_probs, values, saved_ssm = agent.act(
        screen=screen,
        minimap=minimap,
        entities=entities,
        entity_mask=entity_mask,
        scalars=scalars,
        avail_actions=avail_actions
    )
    
    print("Decision actions sampled:")
    for k, v in actions.items():
        print(f"  - {k:<12} : {v}")
        
    print("Log probabilities:")
    for k, v in log_probs.items():
        print(f"  - {k:<12} : {v:.4f}")
        
    print(f"Critic State-Value Estimates -> Macro V(s): {values[0]:.4f} | Micro V(s): {values[1]:.4f}")
    print(f"SSM State shape: {saved_ssm.shape}")
    
    # ----------------------------------------------------
    # Verification 1: Legal Action Mask Check
    # ----------------------------------------------------
    print("\n--- Verification 1: Dynamic Action Masking Check ---")
    assert actions["a_major"] in legal_action_indices, f"FAILED: Chosen action {actions['a_major']} is NOT in legal actions {legal_action_indices}!"
    print(f"SUCCESS: Sampled action {actions['a_major']} is legally valid (belongs to {legal_action_indices})!")
    
    # ----------------------------------------------------
    # Verification 2: Pointer Network Unit Selector Check
    # ----------------------------------------------------
    print("\n--- Verification 2: Pointer Network Padding Check ---")
    # Verify that we never sample a padded entity (indices 88 to 127 are marked as padded)
    assert actions["a_unit"] < 88, f"FAILED: Selected padded entity index {actions['a_unit']} (max valid index is 87)!"
    print(f"SUCCESS: Selected entity index {actions['a_unit']} is within valid unpadded boundaries (< 88)!")
    
    # ----------------------------------------------------
    # Verification 3: Buffer sequence insertion and PPO Update
    # ----------------------------------------------------
    print("\n--- Verification 3: Recurrent Rollout buffer & Optimization updates ---")
    print(f"Filling buffer with mock sequence transitions (length={config.rollout_length})...")
    
    agent.reset_memory()
    
    for step in range(config.rollout_length):
        actions_step, log_probs_step, values_step, saved_ssm_step = agent.act(
            screen=screen,
            minimap=minimap,
            entities=entities,
            entity_mask=entity_mask,
            scalars=scalars,
            avail_actions=avail_actions
        )
        
        # Insert transition
        agent.buffer.insert(
            screen=screen,
            minimap=minimap,
            entities=entities,
            entity_mask=entity_mask,
            scalars=scalars,
            avail_actions=avail_actions,
            ssm_state=saved_ssm_step,
            actions=actions_step,
            log_probs=log_probs_step,
            r_macro=0.5,
            r_micro=1.0,
            v_macro=values_step[0],
            v_micro=values_step[1],
            mask=1.0 if step < config.rollout_length - 1 else 0.0
        )
        
    print("Buffer successfully populated!")
    
    print("Computing GAE advantages...")
    agent.buffer.compute_advantages(
        next_val_macro=0.1,
        next_val_micro=0.2,
        next_mask=0.0,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda
    )
    print("GAE advantages computed!")
    
    print("Executing agent.update() recurrent optimization...")
    metrics = agent.update()
    print("Agent optimization complete! Update metrics:")
    for k, v in metrics.items():
        print(f"  - {k:<20} : {v:.6f}")
        
    print("\n" + "=" * 60)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY! SYSTEM STABLE.")
    print("=" * 60)

if __name__ == "__main__":
    main()

