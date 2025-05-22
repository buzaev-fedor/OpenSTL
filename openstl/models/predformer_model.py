import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat

class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., attn_dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        
    def forward(self, x):
        b, n, d = x.shape
        h = self.heads
        
        # Get query, key, value projections
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class MoELayer(nn.Module):
    """Mixture of Experts layer with routing."""
    def __init__(self, dim, num_experts=8, top_k=2, noisy_gate=True, gate_noise=0.1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gate = noisy_gate
        self.gate_noise = gate_noise
        
        # Create gate (router)
        self.gate = nn.Linear(dim, num_experts, bias=False)
        
        # Create experts (each is a feed-forward network)
        self.experts = nn.ModuleList([
            FeedForward(dim, dim * 4) 
            for _ in range(num_experts)
        ])
        
        # Tracking metrics
        self.expert_counts = None
        self.router_prob = None
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Get router outputs
        router_logits = self.gate(x)  # (batch, seq_len, num_experts)
        
        # Add noise for exploration (during training only)
        if self.training and self.noisy_gate:
            noise = torch.randn_like(router_logits) * self.gate_noise
            router_logits = router_logits + noise
        
        # Get router probabilities
        router_probs = nn.functional.softmax(router_logits, dim=-1)
        
        # Get top-k experts per token
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Normalize
        
        # Create mask for each expert
        expert_mask = torch.zeros(
            batch_size, seq_len, self.num_experts, device=x.device
        )
        
        # Track expert counts (for load balancing loss)
        self.expert_counts = torch.zeros(self.num_experts, device=x.device)
        self.router_prob = router_probs.mean(0).mean(0)
        
        # Create output tensor
        final_output = torch.zeros(batch_size, seq_len, d_model, device=x.device)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Create binary mask for current expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
                
            # Count routing decisions
            self.expert_counts[expert_idx] = expert_mask.float().sum()
            
            # Extract inputs for this expert
            expert_inputs = x[expert_mask]
            
            # Process inputs through the expert
            expert_outputs = self.experts[expert_idx](expert_inputs)
            
            # Gather corresponding routing probabilities
            token_position_in_top_k = (top_k_indices == expert_idx).int().argmax(dim=-1)
            routing_probs = torch.gather(
                top_k_probs, -1, token_position_in_top_k.unsqueeze(-1)
            ).squeeze(-1)
            routing_probs = routing_probs[expert_mask]
            
            # Scale outputs by routing probabilities
            expert_outputs = expert_outputs * routing_probs.unsqueeze(-1)
            
            # Add to final output
            final_output[expert_mask] += expert_outputs
            
        return final_output
        
    def get_loss(self):
        """Calculate auxiliary load balancing loss."""
        if self.expert_counts is None or self.router_prob is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        # Compute load balancing loss
        # We want all experts to receive equal number of tokens
        # and router probabilities to be uniform
        balanced_loss = torch.tensor(0.0, device=self.expert_counts.device)
        if self.expert_counts.sum() > 0:
            # Normalize counts
            norm_counts = self.expert_counts / self.expert_counts.sum()
            # Ideal distribution is uniform
            target = torch.ones_like(norm_counts) / self.num_experts
            # Mean squared error between actual and target distributions
            balanced_loss = torch.mean((norm_counts - target) ** 2)
            
            # Also add router probability entropy loss
            router_prob = self.router_prob + 1e-10  # avoid log(0)
            entropy_loss = -(router_prob * torch.log(router_prob)).sum()
            balanced_loss = balanced_loss - 0.1 * entropy_loss  # encourage exploration
            
        return balanced_loss

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., attn_dropout=0., 
                 drop_path=0., use_moe=False, num_experts=8, top_k=2, 
                 noisy_gate=True, gate_noise=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads=heads, dim_head=dim_head, 
                                 dropout=dropout, attn_dropout=attn_dropout)
        self.norm2 = nn.LayerNorm(dim)
        
        # Use either standard FFN or MoE
        self.use_moe = use_moe
        if use_moe:
            self.ff = MoELayer(dim, num_experts=num_experts, top_k=top_k,
                              noisy_gate=noisy_gate, gate_noise=gate_noise)
        else:
            self.ff = FeedForward(dim, dim * 4, dropout=dropout)
            
        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x
        
    def get_moe_loss(self):
        if self.use_moe:
            return self.ff.get_loss()
        return torch.tensor(0.0)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class PredFormer_Model(nn.Module):
    def __init__(self, height, width, num_channels, pre_seq, after_seq, patch_size=8,
                 dim=256, heads=8, dim_head=64, depth=1, Ndepth=6, dropout=0.,
                 attn_dropout=0., drop_path=0., scale_dim=4, use_moe=False,
                 num_experts=8, top_k=2, noisy_gate=True, gate_noise=0.1,
                 moe_loss_weight=0.01, **kwargs):
        super().__init__()
        
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.pre_seq = pre_seq
        self.after_seq = after_seq
        self.patch_size = patch_size
        self.use_moe = use_moe
        self.moe_loss_weight = moe_loss_weight
        
        # Calculate sizes
        h, w = height // patch_size, width // patch_size
        num_patches = h * w
        
        # Patch embedding
        patch_dim = num_channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        # Position embeddings - spatial
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        
        # Position embeddings - temporal
        self.temporal_embedding = nn.Parameter(torch.randn(1, pre_seq, dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                           attn_dropout=attn_dropout, drop_path=drop_path,
                           use_moe=use_moe, num_experts=num_experts, top_k=top_k,
                           noisy_gate=noisy_gate, gate_noise=gate_noise)
            for _ in range(Ndepth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        
        # Project back to patch size
        self.to_pixels = nn.Linear(dim, patch_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, channels, height, width)
        b, t, c, h, w = x.shape
        p = self.patch_size
        
        # Reshape to patches
        x = x.reshape(b, t, c, h // p, p, w // p, p)
        x = x.permute(0, 1, 3, 5, 2, 4, 6)
        x = x.reshape(b, t, (h // p) * (w // p), c * p * p)
        
        # Patch embedding
        x = self.to_patch_embedding(x)  # (b, t, patches, dim)
        
        # Add position embedding to each frame
        x = x + self.pos_embedding
        
        # Add temporal embedding
        x = x + self.temporal_embedding.unsqueeze(2)
        
        # Reshape for transformer blocks
        x = x.reshape(b, t * ((h // p) * (w // p)), -1)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Apply normalization
        x = self.norm(x)
        
        # Reshape back
        x = x.reshape(b, t, (h // p) * (w // p), -1)
        
        # Project back to pixel space
        x = self.to_pixels(x)
        
        # Reshape back to original dimensions but for next frames
        x = x.reshape(b, t, h // p, w // p, c, p, p)
        x = x.permute(0, 1, 4, 2, 5, 3, 6)
        x = x.reshape(b, t, c, h, w)
        
        return x
    
    def get_moe_loss(self):
        """Calculate MoE auxiliary loss from all MoE layers."""
        if not self.use_moe:
            return torch.tensor(0.0)
            
        # Sum losses from all MoE layers
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.transformer_blocks:
            if hasattr(block, 'get_moe_loss'):
                total_loss = total_loss + block.get_moe_loss()
                
        return total_loss 