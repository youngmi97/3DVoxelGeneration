import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelProcessor(nn.Module):
    """Handles downscaling and upscaling of voxel data"""
    def __init__(self, input_size=128, target_size=32):
        super().__init__()
        self.input_size = input_size
        self.target_size = target_size
        
    def downscale(self, x):
        """Downscale voxels from input_size to target_size"""
        scale_factor = self.input_size // self.target_size
        x = F.max_pool3d(x.float(), kernel_size=scale_factor, stride=scale_factor)
        return (x > 0.5).float()
    
    def downsample(self,x):
        scale_factor = self.input_size // self.target_size
        original_shape = x.shape
        new_shape = original_shape[:-3]+(self.target_size, scale_factor, self.target_size, scale_factor, self.target_size, scale_factor)
        x = x.reshape(new_shape)
        x = x.any(dim=(-1)) 
        x = x.any(dim=(-2)) 
        x = x.any(dim=(-3)) 
        return (x > 0.5).float()
    
    def upscale(self, x):
        """Upscale voxels from target_size to input_size"""
        x = F.interpolate(x, size=self.input_size, 
                         mode='trilinear', align_corners=False)
        return (x > 0.5).float()