import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models.dit3d import DiT_S
from models.voxel_processor import VoxelProcessor
from models.diffusion import VoxelDiffusion
from models.augmentation import AugmentedVoxelDataset

def create_dataloader(data, batch_size=32, augment_factor=4, shuffle=True):
    """Create dataloader with augmentation for binary voxel data"""
    # Ensure data is binary
    # data = (data > 0.5).float()
    data = (data > 0.5).float().unsqueeze(1) 
    # Create augmented dataset
    # dataset = AugmentedVoxelDataset(data, augment_factor)
    dataset = torch.utils.data.TensorDataset(data)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4,
        pin_memory=True
    )

def save_samples(model, processor, diffusion, category, epoch):
    """Generate and save binary voxel samples"""
    os.makedirs("samples", exist_ok=True)
    
    # Generate samples
    samples_32 = diffusion.generate_samples(n_samples=32)
    
    # Upscale to 128³
    samples_128 = processor.upscale(samples_32)
    
    # Ensure samples are binary
    samples_128 = (samples_128 > 0.5).float()
    
    # Save samples
    np.save(f"samples/{category}_epoch_{epoch}.npy", samples_128.cpu().numpy())
    
    # Log sample statistics
    wandb.log({
        "num_positive_voxels": samples_128.sum().item() / len(samples_128),
        "sample_variance": samples_128.var().item()
    })

def train(category, n_epochs=100, batch_size=32, device="cuda"):
    """Training pipeline for binary voxel diffusion"""
    # Initialize wandb
    wandb.init(project="voxel-diffusion", name=f"{category}-training-augmented")
    
    # Initialize models
    processor = VoxelProcessor().to(device)
    model = DiT_S(
        input_size=32,
        in_channels=1,
    ).to(device)
    
    diffusion = VoxelDiffusion(model, device=device)
    
    # Load data with augmentation
    train_data = torch.from_numpy(np.load(f"data/hdf5_data/{category}_voxels_train.npy"))
    train_loader = create_dataloader(train_data, batch_size, augment_factor=4)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.CosineAnnealingLR(optimizer, n_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            # batch = batch.to(device)
            batch = batch[0].to(device)
            
            # Ensure input is binary
            batch = (batch > 0.5).float()
            
            # Downscale to 32³
            x_0 = processor.downscale(batch)
            
            # Sample noise and timesteps
            t = torch.randint(0, diffusion.n_steps, (len(x_0),), device=device)
            noise = torch.randn_like(x_0)
            
            # Get noisy samples and predict noise
            x_t = diffusion.q_sample(x_0, t, noise)
            noise_pred = model(x_t, t)
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)
            
            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update stats
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            wandb.log({"batch_loss": loss.item()})
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch stats
        avg_loss = total_loss / len(train_loader)
        wandb.log({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/{category}_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            
            # Generate and save samples
            save_samples(model, processor, diffusion, category, epoch)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    categories = ["chair", "airplane", "table"]
    for category in categories:
        print(f"\nTraining {category} model...")
        train(category, device=device)