import torch
import torch.nn as nn

class VoxelAutoencoder(nn.Module):
    def __init__(self, input_size=64, latent_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(64, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(128, latent_dim, 4, stride=2, padding=1),
            nn.GELU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon