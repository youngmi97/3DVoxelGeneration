# models/augmentation.py

import torch
import torch.nn as nn

class VoxelAugmenter:
    """3D voxel data augmentation"""
    def __init__(self, device="cuda"):
        self.device = device
        
    def random_rotate90(self, voxel):
        """Randomly rotate 90 degrees along any axis"""
        # For 3D data, dims are [C, D, H, W] or just [D, H, W]
        k = torch.randint(0, 4, (1,)).item()  # 0-3 times
        if len(voxel.shape) == 3:  # [D, H, W]
            axis = torch.randint(0, 3, (1,)).item()  # axes 0,1,2
            dims = [(0,1), (1,2), (2,0)][axis]
        else:  # [C, D, H, W]
            axis = torch.randint(0, 3, (1,)).item()  # axes 1,2,3
            dims = [(1,2), (2,3), (3,1)][axis]
        return torch.rot90(voxel, k, dims)
    
    def random_flip(self, voxel):
        """Randomly flip along any axis"""
        # For 3D data, handle both [C, D, H, W] and [D, H, W] formats
        dims = list(range(len(voxel.shape)))
        if len(dims) == 4:  # [C, D, H, W]
            dims = dims[1:]  # Skip channel dimension
            
        for axis in dims:
            if torch.rand(1).item() > 0.5:
                voxel = torch.flip(voxel, [axis])
        return voxel
    
    def __call__(self, voxel):
        """Apply random augmentations"""
        # Ensure input is binary
        voxel = (voxel > 0.5).float()
        
        # Random rotations (50% chance)
        if torch.rand(1).item() > 0.5:
            voxel = self.random_rotate90(voxel)
            
        # Random flips (50% chance)
        if torch.rand(1).item() > 0.5:
            voxel = self.random_flip(voxel)
            
        return voxel

class AugmentedVoxelDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies augmentations"""
    def __init__(self, base_data, augment_factor=4):
        """
        Args:
            base_data: Original voxel data tensor
            augment_factor: How many augmented versions per original sample
        """
        self.base_data = base_data
        self.augment_factor = augment_factor
        self.augmenter = VoxelAugmenter()
        self.length = len(base_data) * augment_factor
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Get base sample
        base_idx = idx // self.augment_factor
        base_sample = self.base_data[base_idx]
        
        # Skip augmentation for original samples
        if idx % self.augment_factor == 0:
            return base_sample.unsqueeze(0)
            
        # Apply augmentations
        return self.augmenter(base_sample).unsqueeze(0)