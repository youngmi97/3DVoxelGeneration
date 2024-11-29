import torch
import numpy as np
from models.diffusion import VoxelDiffusion
from models.dit3d import DiT_S
from models.voxel_processor import VoxelProcessor

# Generate 1000 samples given checkpoints for df and sr
def generate_samples(category, n_samples=1000, batch_size=32, device="cuda"):
    model = DiT_S(input_size=32).to(device)
    processor = VoxelProcessor(input_size=64, target_size=32).to(device)
    diffusion = VoxelDiffusion(model, device=device)

    # Load checkpoints
    diffusion_ckpt = torch.load(f"checkpoints/S_32_{category}/{category}_epoch_1000.pt")
    model.load_state_dict(diffusion_ckpt['model_state_dict'])
    sr_ckpt = torch.load(f"checkpoints/SR/sr_{category}_epoch_v1.pt")
    processor.sr_module.load_state_dict(sr_ckpt['model_state_dict'])

    all_samples = []
    model.eval()
    processor.sr_module.eval()
    
    for i in range(0, n_samples, batch_size):
        batch_size = min(batch_size, n_samples - i)
        with torch.no_grad():
            samples_32 = diffusion.generate_samples(n_samples=batch_size, shape=(32, 32, 32))
            samples_64 = processor.upscale(samples_32)
            # all_samples.append(samples_64.cpu())
            all_samples.append(samples_64.squeeze(1).cpu())
            torch.cuda.empty_cache()

    final_samples = torch.cat(all_samples, dim=0)
    np.save(f"samples/{category}_final_samples.npy", final_samples.numpy())

if __name__ == "__main__":
   import sys
   category = sys.argv[1]
   generate_samples(category)