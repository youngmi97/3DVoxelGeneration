import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse

from models.dit3d import DiT_S, DiT_L_4, DiT_M_4
from models.voxel_processor import VoxelProcessor
from models.diffusion import VoxelDiffusion

def create_dataloader(data, batch_size=16, augment_factor=4, shuffle=True):
    """Create dataloader with augmentation for binary voxel data"""
    data = (data > 0.5).float().unsqueeze(1) 
    dataset = torch.utils.data.TensorDataset(data)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=4,
        pin_memory=True
    )

def save_samples(model, processor, diffusion, category, epoch, input_size=32):
    """Generate and save binary voxel samples"""
    os.makedirs("samples", exist_ok=True)

    # Load SR checkpoint
    checkpoint = torch.load(f'checkpoints/SR/sr_{category}_epoch_v1.pt')
    processor.sr_module.load_state_dict(checkpoint['model_state_dict'])
    processor.sr_module.eval()
    
    with torch.no_grad():
        samples_32 = diffusion.generate_samples(n_samples=32, shape=(input_size,)*3)
        samples_64 = processor.upscale(samples_32)
    
    np.save(f"samples/{category}_epoch_{epoch}.npy", samples_64.cpu().numpy())
    wandb.log({
        "num_positive_voxels": samples_64.sum().item() / len(samples_64),
        "sample_variance": samples_64.var().item()
    })

def train_diffusion(category, n_epochs=1000, batch_size=16, input_size=32, device="cuda", resume_from=None):
    """Training pipeline for diffusion model"""
    wandb.init(project="voxel-diffusion", name=f"{category}-diffusion-training", resume="allow")
    
    processor = VoxelProcessor(target_size=input_size).to(device)
    model = DiT_S(
        input_size=input_size,
        in_channels=1,
    ).to(device)
    
    diffusion = VoxelDiffusion(model, device=device)
    
    train_data = torch.from_numpy(np.load(f"data/hdf5_data/{category}_voxels_train.npy"))
    train_loader = create_dataloader(train_data, batch_size, augment_factor=4)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    start_epoch = 0

    if resume_from:
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    os.makedirs("checkpoints", exist_ok=True)
    
    try:
        for epoch in range(start_epoch, n_epochs):
            model.train()
            total_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
            for batch in pbar:
                batch = batch[0].to(device)
                batch = (batch > 0.5).float()
                x_0 = processor.downscale(batch)
                
                t = torch.randint(0, diffusion.n_steps, (len(x_0),), device=device)
                noise = torch.randn_like(x_0)
                
                x_t = diffusion.q_sample(x_0, t, noise)
                noise_pred = model(x_t, t)
                
                loss = F.mse_loss(noise_pred, noise)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
                wandb.log({"batch_loss": loss.item()})
            
            scheduler.step()
            
            avg_loss = total_loss / len(train_loader)
            wandb.log({
                "epoch": epoch,
                "avg_loss": avg_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
            if (epoch + 1) % 20 == 0:
                checkpoint_path = f"checkpoints/DF/{category}_epoch_{epoch+1}.pt"
                os.makedirs("checkpoints/DF", exist_ok=True)
                try:
                    temp_checkpoint_path = checkpoint_path + '.tmp'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': avg_loss,
                    }, temp_checkpoint_path)
                    
                    os.replace(temp_checkpoint_path, checkpoint_path)
                    save_samples(model, processor, diffusion, category, epoch, input_size=input_size)
                except Exception as e:
                    print(f"Error saving checkpoint: {e}")
                    torch.cuda.empty_cache()
                    continue
                
    except Exception as e:
        print(f"Training interrupted: {e}")
        emergency_checkpoint_path = f"checkpoints/DF/{category}_emergency_epoch_{epoch}.pt"
        os.makedirs("checkpoints/DF", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, emergency_checkpoint_path)
        print(f"Saved emergency checkpoint to {emergency_checkpoint_path}")

def train_sr(category, sr_epochs=100, batch_size=16, input_size=32, device="cuda"):
    """Training pipeline for super resolution module"""
    wandb.init(project="voxel-diffusion", name=f"{category}-sr-training", resume="allow")
    
    processor = VoxelProcessor(target_size=input_size).to(device)
    train_data = torch.from_numpy(np.load(f"data/hdf5_data/{category}_voxels_train.npy"))
    train_loader = create_dataloader(train_data, batch_size, augment_factor=4)
    
    sr_optimizer = torch.optim.AdamW(processor.sr_module.parameters(), lr=1e-4)
    sr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(sr_optimizer, sr_epochs)
    
    os.makedirs("checkpoints/SR", exist_ok=True)
    
    for epoch in range(sr_epochs):
        total_sr_loss = 0
        pbar = tqdm(train_loader, desc=f"SR Epoch {epoch+1}/{sr_epochs}")
        
        for batch in pbar:
            batch = batch[0].to(device)
            sr_loss = processor.train_sr(batch, sr_optimizer, epoch=epoch, category=category)
            total_sr_loss += sr_loss
            pbar.set_postfix({"sr_loss": sr_loss})
            wandb.log({"sr_batch_loss": sr_loss})
        
        sr_scheduler.step()
        avg_sr_loss = total_sr_loss / len(train_loader)
        wandb.log({
            "sr_epoch": epoch,
            "avg_sr_loss": avg_sr_loss,
            "sr_learning_rate": sr_scheduler.get_last_lr()[0]
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train voxel diffusion or super resolution model')
    parser.add_argument('--mode', type=str, required=True, choices=['sr', 'df'],
                      help='Training mode: sr (super resolution) or df (diffusion)')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--input_size', type=int, default=32,
                      help='Input size for voxels')
    parser.add_argument('--resume_from', type=str, default=None,
                      help='Path to checkpoint to resume training from (diffusion mode only)')
    parser.add_argument('--categories', nargs='+', default=['airplane'],
                      help='Categories to train on')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for category in args.categories:
        print(f"\nTraining {category} model...")
        if args.mode == 'sr':
            train_sr(category, batch_size=args.batch_size, 
                    input_size=args.input_size, device=device)
        else:
            train_diffusion(category, batch_size=args.batch_size,
                          input_size=args.input_size, device=device,
                          resume_from=args.resume_from)