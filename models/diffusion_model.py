import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import math
import numpy as np
from .transformer_layers import DiT3DBlock

def gaussian_kl(mean1, var1, mean2, var2):
    """Compute KL divergence between two Gaussians"""
    return 0.5 * (
        torch.log(var2) - torch.log(var1) + 
        (var1 + (mean1 - mean2)**2) / var2 - 1
    ).mean()

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

        # Embeddings
        self.time_embed = TimestepEmbedder(dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 512, dim))  # 8x8x8 = 512 positions
        
        # Noise scheduler
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.num_timesteps = 1000
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        self.register_buffer('betas', betas)
        alphas = 1. - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        
        # Main transformer blocks with window attention
        self.noise_predictor = nn.ModuleList([
            DiT3DBlock(
                dim=dim,
                num_heads=8,
                window_size=window_size
            ) for _ in range(num_layers)
        ])
        
        # Variance prediction network
        self.var_predictor = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Linear and devoxelization layers
        self.final_proj = nn.Linear(dim, 3)  # Project to point cloud coordinates
        
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize position embeddings
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Initialize variance predictor
        for module in self.var_predictor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod[t])
        
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise, noise
    
    def predict_start_from_noise(self, x_t, t, predicted_noise):
        """Predict x_0 from noise"""
        alpha_cumprod = self.alphas_cumprod[t]
        return (x_t - torch.sqrt(1. - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
    
    def forward(self, x, t):
        B = x.shape[0]
        
        # Get latent representation
        with torch.no_grad():
            latent = self.autoencoder.encoder(x)
        
        # Add noise
        noisy_latent, target_noise = self.q_sample(latent, t)
        
        # Add positional embeddings
        h = noisy_latent.reshape(B, 512, -1) + self.pos_embed
        t_emb = self.time_embed(t)
        
        # Window attention transformer blocks
        for block in self.noise_predictor:
            h = block(h, t_emb)
        
        # Predict noise and variance
        predicted_noise = h.mean(dim=1)
        predicted_var = self.var_predictor(h).mean(dim=1)
        
        # Calculate losses
        noise_loss = F.mse_loss(predicted_noise, target_noise)
        
        # Calculate KL divergence
        true_mean = self.predict_start_from_noise(noisy_latent, t, target_noise)
        pred_mean = self.predict_start_from_noise(noisy_latent, t, predicted_noise)
        
        kl_loss = gaussian_kl(
            mean1=pred_mean, 
            var1=predicted_var,
            mean2=true_mean, 
            var2=torch.ones_like(predicted_var) * self.beta_end
        )
        
        # Combined loss
        loss = noise_loss + 0.001 * kl_loss
        
        return loss

# class ImprovedLatentDiffusionModel(nn.Module):
#     def __init__(self, autoencoder, num_layers=8, window_size=4):
#         super().__init__()
#         dim = 256
#         self.autoencoder = autoencoder
#         for param in self.autoencoder.parameters():
#             param.requires_grad = False
        
#         self.time_embed = TimestepEmbedder(dim)
#         self.window_size = window_size
        
#         # Position embeddings for 8x8x8 latent space
#         self.pos_embed = nn.Parameter(torch.zeros(1, 512, dim))
#         nn.init.normal_(self.pos_embed, std=0.02)
        
#         self.noise_predictor = nn.ModuleList([
#             DiT3DBlock(dim, num_heads=8, window_size=window_size)
#             for _ in range(num_layers)
#         ])
    
#     def forward(self, x, t):
#         B = x.shape[0]
        
#         # Get latent representation
#         with torch.no_grad():
#             latent = self.autoencoder.encoder(x)  # [B, 256, 8, 8, 8]
        
#         # Add noise
#         t_emb = self.time_embed(t)
#         noise = torch.randn_like(latent)
#         alpha_t = self.get_alpha(t)
#         noisy_latent = torch.sqrt(alpha_t) * latent + torch.sqrt(1 - alpha_t) * noise
        
#         # Reshape to sequence
#         h = noisy_latent.flatten(2).transpose(1, 2)  # [B, 512, 256]
        
#         # Add position embeddings
#         h = h + self.pos_embed
        
#         # Pass through transformer blocks
#         for block in self.noise_predictor:
#             h = block(h, t_emb)
        
#         # Reshape back
#         h = h.transpose(1, 2).reshape_as(noisy_latent)
#         return F.mse_loss(h, noise)
    
#     def get_alpha(self, t, beta_start=1e-4, beta_end=0.02):
#         beta_t = beta_start + t * (beta_end - beta_start)
#         alpha_t = 1 - beta_t
#         alpha = torch.cumprod(alpha_t, dim=0)
#         return alpha.view(-1, 1, 1, 1, 1)
    
#     @torch.no_grad()
#     def sample(self, batch_size=8, steps=100):
#         device = next(self.parameters()).device
#         latent = torch.randn(batch_size, 256, 8, 8, 8).to(device)
        
#         for t in reversed(range(steps)):
#             t_tensor = torch.ones(batch_size).to(device) * t / steps
#             t_emb = self.time_embed(t_tensor)
#             alpha_t = self.get_alpha(t_tensor)
            
#             # Reshape to sequence
#             h = latent.flatten(2).transpose(1, 2)
#             h = h + self.pos_embed
            
#             for block in self.noise_predictor:
#                 h = block(h, t_emb)
            
#             h = h.transpose(1, 2).reshape_as(latent)
            
#             latent = 1 / torch.sqrt(alpha_t) * (latent - (1 - alpha_t) / torch.sqrt(1 - alpha_t) * h)
#             if t > 0:
#                 latent = latent + torch.sqrt(1 - alpha_t) * torch.randn_like(latent)
        
#         voxels = self.autoencoder.decoder(latent)
#         return F.interpolate(voxels, size=(128, 128, 128), mode='trilinear')

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