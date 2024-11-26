import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

def generate_evaluation_samples(model, category, num_samples=200, batch_size=2):  # Reduced samples and batch size
    device = next(model.parameters()).device
    save_path = f'generated_samples/{category}_samples.npy'
    
    all_samples = np.zeros((num_samples, 128, 128, 128), dtype=np.float32)
    
    print(f"[*] Generating {num_samples} samples in batches of {batch_size}...")
    for i in tqdm(range(0, num_samples, batch_size)):
        curr_batch = min(batch_size, num_samples - i)
        with torch.cuda.amp.autocast():  # Use mixed precision
            batch_samples = model.sample(batch_size=curr_batch)
            batch_samples = F.interpolate(batch_samples, size=(128, 128, 128), mode='trilinear')
        
        batch_samples = (batch_samples.squeeze(1).cpu() > 0.5).float().numpy()
        all_samples[i:i+curr_batch] = batch_samples
        
        # Clear cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()
    
    print(f"[*] Saving samples to {save_path}")
    np.save(save_path, all_samples)
    return save_path