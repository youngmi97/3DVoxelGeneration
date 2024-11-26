import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import math
import numpy as np
from .transformer_layers import DiT3DBlock


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations"""
    def __init__(self, dim, freq_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.freq_dim = freq_dim

    def forward(self, t):
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half) / half).to(t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding)

class ImprovedLatentDiffusionModel(nn.Module):
    def __init__(self, autoencoder, num_layers=8, window_size=4):
        super().__init__()
        dim = 256
        self.autoencoder = autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        
        self.time_embed = TimestepEmbedder(dim)
        self.window_size = window_size
        
        # Position embeddings for 8x8x8 latent space
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.noise_predictor = nn.ModuleList([
            DiT3DBlock(dim, num_heads=8, window_size=window_size)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, t):
        B = x.shape[0]
        
        # Get latent representation
        with torch.no_grad():
            latent = self.autoencoder.encoder(x)  # [B, 256, 8, 8, 8]
        
        # Add noise
        t_emb = self.time_embed(t)
        noise = torch.randn_like(latent)
        alpha_t = self.get_alpha(t)
        noisy_latent = torch.sqrt(alpha_t) * latent + torch.sqrt(1 - alpha_t) * noise
        
        # Reshape to sequence
        h = noisy_latent.flatten(2).transpose(1, 2)  # [B, 512, 256]
        
        # Add position embeddings
        h = h + self.pos_embed
        
        # Pass through transformer blocks
        for block in self.noise_predictor:
            h = block(h, t_emb)
        
        # Reshape back
        h = h.transpose(1, 2).reshape_as(noisy_latent)
        return F.mse_loss(h, noise)
    
    def get_alpha(self, t, beta_start=1e-4, beta_end=0.02):
        beta_t = beta_start + t * (beta_end - beta_start)
        alpha_t = 1 - beta_t
        alpha = torch.cumprod(alpha_t, dim=0)
        return alpha.view(-1, 1, 1, 1, 1)
    
    @torch.no_grad()
    def sample(self, batch_size=8, steps=100):
        device = next(self.parameters()).device
        latent = torch.randn(batch_size, 256, 8, 8, 8).to(device)
        
        for t in reversed(range(steps)):
            t_tensor = torch.ones(batch_size).to(device) * t / steps
            t_emb = self.time_embed(t_tensor)
            alpha_t = self.get_alpha(t_tensor)
            
            # Reshape to sequence
            h = latent.flatten(2).transpose(1, 2)
            h = h + self.pos_embed
            
            for block in self.noise_predictor:
                h = block(h, t_emb)
            
            h = h.transpose(1, 2).reshape_as(latent)
            
            latent = 1 / torch.sqrt(alpha_t) * (latent - (1 - alpha_t) / torch.sqrt(1 - alpha_t) * h)
            if t > 0:
                latent = latent + torch.sqrt(1 - alpha_t) * torch.randn_like(latent)
        
        voxels = self.autoencoder.decoder(latent)
        return F.interpolate(voxels, size=(128, 128, 128), mode='trilinear')

def get_3d_position_embeddings(height, width, depth, dim):
    total_positions = height * width * depth
    position_enc = torch.arange(dim) / dim
    div_term = torch.exp(-math.log(10000.0) * position_enc)
    
    pos_embed = torch.zeros(total_positions, dim)
    position = torch.arange(total_positions)
    
    # Calculate positions in 3D space
    z = position % depth
    y = (position // depth) % width
    x = position // (depth * width)
    
    # Generate embeddings
    for pos_idx, (xi, yi, zi) in enumerate(zip(x, y, z)):
        sin_inp = pos_idx * div_term
        pos_embed[pos_idx] = torch.sin(sin_inp)
    
    return pos_embed