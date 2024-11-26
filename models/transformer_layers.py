import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# def window_partition(x, window_size):
#     """Convert input to windows"""
#     B, H, W, D, C = x.shape
    
#     # Pad if needed
#     pad_h = (window_size - H % window_size) % window_size
#     pad_w = (window_size - W % window_size) % window_size
#     pad_d = (window_size - D % window_size) % window_size
    
#     if pad_h > 0 or pad_w > 0 or pad_d > 0:
#         x = F.pad(x, (0, 0, 0, pad_d, 0, pad_w, 0, pad_h))
    
#     # Reshape to windows
#     x = x.view(B, H//window_size, window_size, W//window_size, 
#                window_size, D//window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
#     windows = windows.view(-1, window_size**3, C)
#     return windows

# def window_reverse(windows, window_size, H, W, D):
#     """Convert windows back to original shape"""
#     B = int(windows.shape[0] / (H * W * D / window_size**3))
#     x = windows.view(B, H//window_size, W//window_size, D//window_size, 
#                     window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
#     x = x.view(B, H, W, D, -1)
#     return x

class Attention3D(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=4, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DiT3DBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=4, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention3D(dim, num_heads, window_size)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )

    def forward(self, x, c):
        B, L, C = x.shape
        H = W = D = 8  # Fixed for 8x8x8 latent space
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Reshape for window attention
        x_reshaped = x.view(B, H, W, D, C)
        
        # Apply attention with modulation
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h = self.attn(h)
        x = x + gate_msa.unsqueeze(1) * h
        
        # MLP with modulation
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * h
        
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)