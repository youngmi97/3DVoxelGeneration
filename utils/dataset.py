import torch
from torch.utils.data import Dataset
import numpy as np
import os

class ShapeNetDataset(Dataset):
    def __init__(self, data_dir, category, split='train'):
        self.data_path = os.path.join(data_dir, 'hdf5_data', f'{category}_voxels_{split}.npy')
        self.data = np.load(self.data_path)
        self.data = torch.from_numpy(self.data).unsqueeze(1).float()  # [B, 1, 64, 64, 64]
        self.data = self.data.reshape(32, 4, 32, 4, 32, 4).mean(axis=(1, 3, 5))
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]