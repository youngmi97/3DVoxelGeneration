import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelProcessor(nn.Module):
    """Handles downscaling and upscaling of voxel data"""
    def __init__(self, input_size=128, target_size=32, output_size=64):
        super().__init__()
        self.input_size = input_size
        self.target_size = target_size
        
    def downscale(self, x):
        """Downscale voxels from input_size to target_size"""
        scale_factor = self.input_size // self.target_size
        x = F.max_pool3d(x.float(), kernel_size=scale_factor, stride=scale_factor)
        return (x > 0.5).float()
    
    def upscale(self, x):
        """Upscale voxels from target_size to input_size"""
        x = F.interpolate(x, size=self.output_size, 
                         mode='trilinear', align_corners=False)
        return (x > 0.5).float()