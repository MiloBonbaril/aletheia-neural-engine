import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List

# Orthogonal layer initialization (standard for high-stability RL)
def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ResNetBlock(nn.Module):
    """
    Light residual convolutional block for processing spatial SC2 features.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class SpatialTorso(nn.Module):
    """
    CNN processing topological screen (84x84) and minimap (64x64) representations.
    """
    def __init__(self, screen_ch: int, minimap_ch: int, out_dim: int = 128):
        super().__init__()
        # Screen Processing (Input: 84x84)
        self.screen_conv = nn.Sequential(
            nn.Conv2d(screen_ch, 16, kernel_size=5, stride=2, padding=2), # -> 42x42
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResNetBlock(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),       # -> 21x21
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResNetBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),       # -> 11x11
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                                # -> 1x1
            nn.Flatten()
        )
        
        # Minimap Processing (Input: 64x64)
        self.minimap_conv = nn.Sequential(
            nn.Conv2d(minimap_ch, 16, kernel_size=5, stride=2, padding=2), # -> 32x32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            ResNetBlock(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),       # -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResNetBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),       # -> 8x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                                # -> 1x1
            nn.Flatten()
        )
        
        # Output Projection
        self.proj = layer_init(nn.Linear(64 + 64, out_dim))

    def forward(self, screen: torch.Tensor, minimap: torch.Tensor) -> torch.Tensor:
        s_feat = self.screen_conv(screen)
        m_feat = self.minimap_conv(minimap)
        combined = torch.cat([s_feat, m_feat], dim=-1)
        return F.relu(self.proj(combined))

class EntityTorso(nn.Module):
    """
    Self-Attention block to digest variable-length lists of units.
    """
    def __init__(self, in_features: int, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.emb_layer = layer_init(nn.Linear(in_features, embed_dim))
        
        # Self-Attention Blocks
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            layer_init(nn.Linear(embed_dim, embed_dim * 2)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim * 2, embed_dim))
        )

    def forward(self, entities: torch.Tensor, padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            entities: tensor of shape (B, N, F_e)
            padding_mask: bool tensor of shape (B, N) where True means padded/ignored
        Returns:
            entity_embeddings: (B, N, embed_dim)
            pooled_entities: (B, embed_dim)
        """
        # Embed entities
        x = F.relu(self.emb_layer(entities)) # (B, N, embed_dim)
        
        # Multi-Head Attention
        attn_out, _ = self.mha(x, x, x, key_padding_mask=padding_mask)
        x = self.ln1(x + attn_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out) # (B, N, embed_dim)
        
        # Masked average pooling
        # Convert mask to float where 1.0 is a real unit, 0.0 is padded
        valid_mask = (~padding_mask).float().unsqueeze(-1) # (B, N, 1)
        sum_emb = (x * valid_mask).sum(dim=1)
        num_valid = valid_mask.sum(dim=1).clamp(min=1.0)
        pooled = sum_emb / num_valid # (B, embed_dim)
        
        return x, pooled

class ScalarTorso(nn.Module):
    """
    MLP that processes global macro economics values.
    """
    def __init__(self, in_features: int, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(in_features, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, out_dim)),
            nn.ReLU()
        )

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        return self.net(scalars)

class MambaSSM(nn.Module):
    """
    A pure-PyTorch selective State Space Model (SSM) layer.
    Approximates Mamba's input-dependent scan (selection mechanism).
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        
        # Dynamic projections (Selective SSM)
        self.in_proj = layer_init(nn.Linear(d_model, self.d_inner * 2, bias=False))
        
        # Sequence-dimension Local Convolution
        self.conv = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )
        
        # Parameter matrices
        self.x_proj = layer_init(nn.Linear(self.d_inner, d_state * 2 + d_model, bias=False))
        self.dt_proj = layer_init(nn.Linear(d_model, self.d_inner, bias=True))
        
        # Continuous state matrix A
        A = torch.arange(1, self.d_inner + 1).float().unsqueeze(-1).expand(-1, d_state)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = layer_init(nn.Linear(self.d_inner, d_model, bias=False))

    def step(self, x: torch.Tensor, ssm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step-by-step recurrent scan (used in environment loop).
        Args:
            x: (B, D_model)
            ssm_state: (B, D_inner, D_state)
        Returns:
            out: (B, D_model)
            new_ssm_state: (B, D_inner, D_state)
        """
        # Project inputs
        projected = self.in_proj(x)
        x_branch, gate = projected.chunk(2, dim=-1)
        
        # Local conv fallback (for single step: identity or small buffer)
        # selective parameter projection
        x_proj_out = self.x_proj(x_branch)
        dt, B, C = x_proj_out.split([self.d_model, self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(dt)) # (B, D_inner)
        
        # Discretize matrices
        A = -torch.exp(self.A_log) # (D_inner, D_state)
        
        # Discretize A
        # dA shape: (B, D_inner, D_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        
        # dB shape: (B, D_inner, D_state)
        dB = dt.unsqueeze(-1) * B.unsqueeze(1)
        
        # Update state
        # ssm_state: (B, D_inner, D_state)
        # x_branch: (B, D_inner)
        new_state = dA * ssm_state + dB * x_branch.unsqueeze(-1)
        
        # Compute Output
        # C: (B, D_state)
        y = torch.bmm(new_state, C.unsqueeze(-1)).squeeze(-1) # (B, D_inner)
        y = y + x_branch * self.D.unsqueeze(0)
        
        # Multiplicative gating
        y = y * F.silu(gate)
        
        out = self.out_proj(y)
        return out, new_state

    def forward(self, x: torch.Tensor, ssm_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequential scan (used in sequential training).
        Args:
            x: (B, L, D_model)
            ssm_state: (B, D_inner, D_state) - initial state
        """
        B, L, _ = x.shape
        outs = []
        curr_state = ssm_state
        for t in range(L):
            xt = x[:, t, :]
            ot, curr_state = self.step(xt, curr_state)
            outs.append(ot.unsqueeze(1))
        
        return torch.cat(outs, dim=1), curr_state

class CausalSelfAttention(nn.Module):
    """
    Self-Attention with a rolling history buffer or causal masking.
    """
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            layer_init(nn.Linear(embed_dim, embed_dim * 2)),
            nn.ReLU(),
            layer_init(nn.Linear(embed_dim * 2, embed_dim))
        )

    def forward(self, x: torch.Tensor, history: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, D_model)
            history: Optional cache of past hidden states (B, L_hist, D_model)
        """
        B, L, D = x.shape
        
        if history is not None and history.shape[1] > 0:
            # Step mode: attend to full history + new token
            full_seq = torch.cat([history, x], dim=1) # (B, L_hist + 1, D)
            attn_out, _ = self.mha(x, full_seq, full_seq)
            new_history = full_seq.detach() # Save for next step
        else:
            # Training mode: attend with causal mask
            mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
            attn_out, _ = self.mha(x, x, x, attn_mask=mask, is_causal=True)
            new_history = x.detach()
            
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        
        return x, new_history

class MambaAttentionCore(nn.Module):
    """
    State-of-the-art hybrid core combining linear Selective SSM (Mamba)
    for sequence streaming, and Causal Attention for precise long-range recall.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        
        # Mamba SSM layer
        self.mamba = MambaSSM(d_model, d_state=d_state, expand=expand)
        self.ln_mamba = nn.LayerNorm(d_model)
        
        # Attention Layer
        self.attention = CausalSelfAttention(d_model, num_heads=num_heads)
        self.ln_attn = nn.LayerNorm(d_model)

    def forward(
        self, 
        x: torch.Tensor, 
        ssm_state: torch.Tensor, 
        attn_history: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: fused features (B, L, D_model)
            ssm_state: recurrent state for Mamba (B, D_inner, D_state)
            attn_history: cache for self-attention (B, L_hist, D_model)
        Returns:
            out: representation vector (B, L, D_model)
            new_ssm_state: updated SSM state (B, D_inner, D_state)
            new_attn_history: updated Attention cache (B, L_hist+L, D_model)
        """
        # 1. Process via Mamba SSM
        mamba_out, new_ssm_state = self.mamba(x, ssm_state)
        mamba_out = self.ln_mamba(mamba_out)
        
        # 2. Process via Causal Attention
        core_out, new_attn_history = self.attention(mamba_out, attn_history)
        core_out = self.ln_attn(core_out)
        
        return core_out, new_ssm_state, new_attn_history

class MacroPolicy(nn.Module):
    """
    Macro-Manager issuing high-level strategic commands (strategic goal 'g').
    """
    def __init__(self, d_model: int, num_macro_goals: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(d_model, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, num_macro_goals), std=0.01)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, L, d_model) -> output (B, L, num_macro_goals)
        return self.net(x)

class MicroPolicy(nn.Module):
    """
    Micro-Manager executing frame-by-frame autoregressive tactical actions.
    Uses dynamic PySC2 Action Masking and Pointer Networks.
    """
    def __init__(
        self, 
        d_model: int, 
        num_macro_goals: int, 
        goal_embed_dim: int,
        num_actions: int,
        spatial_resolution: int = 64,
        entity_embed_dim: int = 128
    ):
        super().__init__()
        self.spatial_res = spatial_resolution
        
        # Macro goal embedding
        self.goal_emb = nn.Embedding(num_macro_goals, goal_embed_dim)
        
        # Combined feature dimensionality
        combined_dim = d_model + goal_embed_dim
        
        # Head 1: PySC2 Action Head
        self.action_head = layer_init(nn.Linear(combined_dim, num_actions), std=0.01)
        self.action_emb = nn.Embedding(num_actions, 64)
        
        # Head 2: Spatial Head (X, Y coords)
        spatial_input_dim = combined_dim + 64
        self.x_head = layer_init(nn.Linear(spatial_input_dim, spatial_resolution), std=0.01)
        self.y_head = layer_init(nn.Linear(spatial_input_dim, spatial_resolution), std=0.01)
        
        # Head 3: Unit Pointer Network Query
        # Spatial coordinate embedding
        self.x_emb = nn.Embedding(spatial_resolution, 32)
        self.y_emb = nn.Embedding(spatial_resolution, 32)
        
        pointer_input_dim = spatial_input_dim + 64 # combined_dim + action_emb + x_emb + y_emb
        self.pointer_query_net = nn.Sequential(
            layer_init(nn.Linear(pointer_input_dim, entity_embed_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(entity_embed_dim, entity_embed_dim))
        )

    def forward(
        self,
        core_features: torch.Tensor,        # (B, L, d_model)
        macro_goal: torch.Tensor,           # (B, L)
        available_actions_mask: torch.Tensor, # (B, L, num_actions) bool mask: True if action is legal
        entity_embeddings: torch.Tensor,     # (B, L, N, entity_embed_dim)
        entity_padding_mask: torch.Tensor,   # (B, L, N) bool mask: True if padded
        actions_taken: Optional[Dict[str, torch.Tensor]] = None # For log-prob computation during updates
    ) -> Dict[str, torch.Tensor]:
        """
        Runs hierarchical autoregressive decision-making.
        """
        B, L, _ = core_features.shape
        
        # Embed macro goal
        g_emb = self.goal_emb(macro_goal) # (B, L, goal_embed_dim)
        
        # Combine Core state and Macro Goal
        f_micro = torch.cat([core_features, g_emb], dim=-1) # (B, L, combined_dim)
        
        # ----------------------------------------------------
        # Head 1: Major Action Selection with Legal Action Mask
        # ----------------------------------------------------
        action_logits = self.action_head(f_micro) # (B, L, num_actions)
        
        # Apply legal action mask (set prohibited actions to -inf)
        masked_action_logits = action_logits.masked_fill(~available_actions_mask, -1e9)
        
        if actions_taken is not None:
            a_major = actions_taken["major"]
        else:
            probs = F.softmax(masked_action_logits, dim=-1)
            # Sample action sequence or step-by-step
            # For simplicity, flatten temporal dim for categorical sampling
            flat_probs = probs.view(-1, probs.shape[-1])
            a_major = torch.multinomial(flat_probs, 1).view(B, L)
            
        a_major_log_probs = F.log_softmax(masked_action_logits, dim=-1)
        a_major_log_prob = a_major_log_probs.gather(-1, a_major.unsqueeze(-1)).squeeze(-1)
        
        # ----------------------------------------------------
        # Head 2: Spatial Coordinate Selection
        # ----------------------------------------------------
        a_major_emb = self.action_emb(a_major) # (B, L, 64)
        spatial_input = torch.cat([f_micro, a_major_emb], dim=-1) # (B, L, spatial_input_dim)
        
        x_logits = self.x_head(spatial_input) # (B, L, spatial_res)
        y_logits = self.y_head(spatial_input) # (B, L, spatial_res)
        
        if actions_taken is not None:
            a_x = actions_taken["x"]
            a_y = actions_taken["y"]
        else:
            x_probs = F.softmax(x_logits, dim=-1)
            y_probs = F.softmax(y_logits, dim=-1)
            
            a_x = torch.multinomial(x_probs.view(-1, x_probs.shape[-1]), 1).view(B, L)
            a_y = torch.multinomial(y_probs.view(-1, y_probs.shape[-1]), 1).view(B, L)
            
        x_log_probs = F.log_softmax(x_logits, dim=-1).gather(-1, a_x.unsqueeze(-1)).squeeze(-1)
        y_log_probs = F.log_softmax(y_logits, dim=-1).gather(-1, a_y.unsqueeze(-1)).squeeze(-1)
        
        # ----------------------------------------------------
        # Head 3: Unit Pointer Network Selector
        # ----------------------------------------------------
        a_x_emb = self.x_emb(a_x) # (B, L, 32)
        a_y_emb = self.y_emb(a_y) # (B, L, 32)
        
        pointer_input = torch.cat([spatial_input, a_x_emb, a_y_emb], dim=-1) # (B, L, pointer_input_dim)
        
        # Project pointer input to match entity embed dims
        q_pointer = self.pointer_query_net(pointer_input) # (B, L, entity_embed_dim)
        
        # Dot-product over all entity embeddings
        # q_pointer: (B, L, E) -> (B*L, 1, E)
        # entity_embeddings: (B, L, N, E) -> (B*L, N, E)
        flat_q = q_pointer.view(-1, 1, q_pointer.shape[-1])
        flat_keys = entity_embeddings.view(-1, entity_embeddings.shape[-2], entity_embeddings.shape[-1])
        
        # raw pointer logits shape: (B*L, 1, N) -> (B, L, N)
        unit_logits = torch.bmm(flat_keys, flat_q.transpose(1, 2)).view(B, L, -1)
        
        # Apply padding mask on entity options (set padded entries to -inf)
        masked_unit_logits = unit_logits.masked_fill(entity_padding_mask, -1e9)
        
        if actions_taken is not None:
            a_unit = actions_taken["unit"]
        else:
            unit_probs = F.softmax(masked_unit_logits, dim=-1)
            # Clip/clamp to avoid exact zeros in multinom sampling
            flat_u_probs = unit_probs.view(-1, unit_probs.shape[-1]).clamp(min=1e-8)
            a_unit = torch.multinomial(flat_u_probs, 1).view(B, L)
            
        unit_log_probs = F.log_softmax(masked_unit_logits, dim=-1).gather(-1, a_unit.unsqueeze(-1)).squeeze(-1)
        
        return {
            "a_major": a_major,
            "a_x": a_x,
            "a_y": a_y,
            "a_unit": a_unit,
            "log_prob_major": a_major_log_prob,
            "log_prob_x": x_log_probs,
            "log_prob_y": y_log_probs,
            "log_prob_unit": unit_log_probs,
            "action_logits": action_logits,
            "unit_logits": masked_unit_logits
        }

class SC2HierarchicalModel(nn.Module):
    """
    Top-Level Model combining all structural components, Torsos, Core, and Actor-Critics.
    """
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        
        # Torsos
        self.spatial_torso = SpatialTorso(
            screen_ch=config.screen_channels,
            minimap_ch=config.minimap_channels,
            out_dim=config.spatial_embed_dim
        )
        self.entity_torso = EntityTorso(
            in_features=config.entity_features_dim,
            embed_dim=config.entity_embed_dim
        )
        self.scalar_torso = ScalarTorso(
            in_features=config.scalar_features_dim,
            out_dim=config.scalar_embed_dim
        )
        
        # Combined Projection to Mamba Core D_model
        total_in_dim = config.spatial_embed_dim + config.entity_embed_dim + config.scalar_embed_dim
        self.core_proj = layer_init(nn.Linear(total_in_dim, config.d_model))
        
        # Recurrent memory core
        self.core = MambaAttentionCore(
            d_model=config.d_model,
            d_state=config.mamba_d_state,
            expand=config.mamba_expand,
            num_heads=config.transformer_heads
        )
        
        # Macro and Micro policies
        self.macro_policy = MacroPolicy(
            d_model=config.d_model,
            num_macro_goals=config.num_macro_goals
        )
        
        self.micro_policy = MicroPolicy(
            d_model=config.d_model,
            num_macro_goals=config.num_macro_goals,
            goal_embed_dim=config.macro_goal_embed_dim,
            num_actions=config.num_actions,
            spatial_resolution=config.spatial_resolution,
            entity_embed_dim=config.entity_embed_dim
        )
        
        # Critic networks estimating state values V(s) for Macro and Micro PPO branches
        self.macro_critic = layer_init(nn.Linear(config.d_model, 1), std=1.0)
        self.micro_critic = layer_init(nn.Linear(config.d_model, 1), std=1.0)

    def forward(
        self,
        screen: torch.Tensor,                 # (B, L, C_s, 84, 84)
        minimap: torch.Tensor,                # (B, L, C_m, 64, 64)
        entities: torch.Tensor,               # (B, L, N, F_e)
        entity_padding_mask: torch.Tensor,    # (B, L, N) bool mask
        scalars: torch.Tensor,                # (B, L, F_s)
        available_actions_mask: torch.Tensor,  # (B, L, num_actions) bool mask
        ssm_state: torch.Tensor,              # (B, D_inner, D_state) - recurrent memory state
        attn_history: Optional[torch.Tensor] = None, # (B, L_hist, d_model) - transformer context
        macro_goal: Optional[torch.Tensor] = None,   # Given or self-sampled (B, L)
        actions_taken: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Completes a unified forward pass for either step evaluation or rollout trajectory batch optimization.
        """
        B, L, C_s, H_s, W_s = screen.shape
        
        # Flatten Temporal dimensions for processing through feed-forward torsos
        flat_screen = screen.view(B * L, C_s, H_s, W_s)
        flat_minimap = minimap.view(B * L, minimap.shape[-3], minimap.shape[-2], minimap.shape[-1])
        flat_entities = entities.view(B * L, entities.shape[-2], entities.shape[-1])
        flat_entity_mask = entity_padding_mask.view(B * L, entity_padding_mask.shape[-1])
        flat_scalars = scalars.view(B * L, scalars.shape[-1])
        
        # 1. Feed-forward through Torsos
        flat_spatial = self.spatial_torso(flat_screen, flat_minimap)      # (B*L, spatial_embed_dim)
        flat_ent_emb, flat_ent_pool = self.entity_torso(flat_entities, flat_entity_mask) # (B*L, N, embed_dim), (B*L, embed_dim)
        flat_scalar = self.scalar_torso(flat_scalars)                     # (B*L, scalar_embed_dim)
        
        # Unflatten unit structures
        entity_embeddings = flat_ent_emb.view(B, L, flat_ent_emb.shape[-2], flat_ent_emb.shape[-1])
        
        # Concatenate and Project to Core Dimension
        flat_combined = torch.cat([flat_spatial, flat_ent_pool, flat_scalar], dim=-1)
        flat_core_input = self.core_proj(flat_combined)
        core_input = flat_core_input.view(B, L, -1)                       # (B, L, d_model)
        
        # 2. Memory Core Sequence Update
        core_out, new_ssm_state, new_attn_history = self.core(core_input, ssm_state, attn_history) # (B, L, d_model)
        
        # 3. Macro and Micro Decision Logic
        macro_logits = self.macro_policy(core_out) # (B, L, num_macro_goals)
        
        if macro_goal is None:
            # Sample a new macro goal
            macro_probs = F.softmax(macro_logits, dim=-1)
            flat_macro_probs = macro_probs.view(-1, macro_probs.shape[-1])
            macro_goal = torch.multinomial(flat_macro_probs, 1).view(B, L)
            
        macro_log_probs = F.log_softmax(macro_logits, dim=-1)
        macro_log_prob = macro_log_probs.gather(-1, macro_goal.unsqueeze(-1)).squeeze(-1)
        
        # Micro decision execution
        micro_out = self.micro_policy(
            core_features=core_out,
            macro_goal=macro_goal,
            available_actions_mask=available_actions_mask,
            entity_embeddings=entity_embeddings,
            entity_padding_mask=entity_padding_mask,
            actions_taken=actions_taken
        )
        
        # 4. State-Value Estimates V(s)
        macro_values = self.macro_critic(core_out).squeeze(-1) # (B, L)
        micro_values = self.micro_critic(core_out).squeeze(-1) # (B, L)
        
        # Aggregate decisions
        actions = {
            "macro_goal": macro_goal,
            "a_major": micro_out["a_major"],
            "a_x": micro_out["a_x"],
            "a_y": micro_out["a_y"],
            "a_unit": micro_out["a_unit"],
            "log_prob_macro": macro_log_prob,
            "log_prob_major": micro_out["log_prob_major"],
            "log_prob_x": micro_out["log_prob_x"],
            "log_prob_y": micro_out["log_prob_y"],
            "log_prob_unit": micro_out["log_prob_unit"]
        }
        
        return actions, macro_values, micro_values, new_ssm_state, new_attn_history, macro_logits
