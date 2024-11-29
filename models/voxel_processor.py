import torch
import torch.nn as nn
import torch.nn.functional as F
from .super_resolution import VoxelSR

class VoxelProcessor(nn.Module):
   def __init__(self, input_size=64, target_size=32, output_size=64):
       super().__init__()
       self.input_size = input_size
       self.target_size = target_size
       self.output_size = output_size
       self.sr_module = VoxelSR(output_size=output_size)
       
   def downscale(self, x):
       scale_factor = self.input_size // self.target_size
       x = F.max_pool3d(x.float(), kernel_size=scale_factor, stride=scale_factor)
       return (x > 0.5).float()
   
   def upscale(self, x):
       return self.sr_module(x, training=False)
       
   def train_sr(self, high_res, optimizer, epoch, category):
        high_res = high_res.float()
        low_res = self.downscale(high_res)
        sr_pred = self.sr_module(low_res, training=True)
        
        # Calculate weights based on class distribution
        pos_weight = (high_res == 0).float().sum() / (high_res == 1).float().sum()
        loss = F.binary_cross_entropy(sr_pred, high_res, 
                                    weight=torch.where(high_res > 0.5, pos_weight, 1.0))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sr_module.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/SR/sr_{category}_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.sr_module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)

        return loss.item()