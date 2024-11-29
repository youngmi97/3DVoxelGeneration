import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelSR(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=48, output_size=64):
        super().__init__()
        self.output_size = output_size
        self.conv1 = nn.Conv3d(in_channels, hidden_dim, 3, padding=1)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(3)])
        self.final = nn.Sequential(
            nn.Conv3d(hidden_dim, in_channels, 3, padding=1),
            nn.BatchNorm3d(in_channels)
        )
    
    # Original version without tracking number of > 0.5 voxels
    def forward(self, x, training=False):
        x = self.conv1(x)
        orig_features = x.clone()
        
        for block in self.res_blocks:
            x = block(x)
        
        x = x + orig_features  # Skip connection for structure preservation
        x = F.interpolate(x, size=(self.output_size,)*3, mode='nearest')  # Changed to nearest neighbor
        x = torch.sigmoid(self.final(x))
        
        if not training:
            # Adaptive thresholding based on input sparsity
            input_density = x.mean()
            threshold = min(0.5, input_density + 0.1)
            x = (x > threshold).float()
        return x

    # Alternate version tracking > 0.5 voxel count
    # def forward(self, x, training=False):
    #     x = self.conv1(x)
    #     orig_features = x.clone()
        
    #     for block in self.res_blocks:
    #         x = block(x)
        
    #     x = x + orig_features
    #     x = F.interpolate(x, size=(self.output_size,)*3, mode='nearest')
    #     x = torch.sigmoid(self.final(x))
        
    #     if not training:
    #         input_count = torch.sum(x > 0.5)
    #         target_count = input_count // 4  # Since upscaling from 32 to 64 (~8x volume)
    #         sorted_vals = torch.sort(x.flatten(), descending=True)[0]
    #         threshold = sorted_vals[int(target_count)]
    #         x = (x > threshold).float()
    #     return x

class ResidualBlock(nn.Module):
   def __init__(self, channels):
       super().__init__()
       self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
       self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
       self.norm = nn.BatchNorm3d(channels)
       
   def forward(self, x):
       residual = x
       x = F.gelu(self.conv1(self.norm(x)))
       x = self.conv2(self.norm(x))
       return x + residual