import os
import torch
import wandb
from torch.utils.data import DataLoader
from models.autoencoder import VoxelAutoencoder
from models.diffusion_model import ImprovedLatentDiffusionModel
from utils.dataset import ShapeNetDataset
from scripts.train_autoencoder import train_autoencoder
from scripts.train_diffusion import train_diffusion
from scripts.generate_samples import generate_evaluation_samples

def train_and_evaluate(category, batch_size=32, device='cuda'):
    wandb.init(project="3d-voxel-diffusion", name=f"{category}_training")
    
    train_dataset = ShapeNetDataset('data', category, 'train')
    val_dataset = ShapeNetDataset('data', category, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    
    print(f"[*] Training Autoencoder for {category}...")
    # autoencoder = VoxelAutoencoder(input_size=64).to(device)
    # train_autoencoder(autoencoder, train_loader, val_loader, category, device)

    # USE CHECKPOINT
    autoencoder = VoxelAutoencoder(input_size=64).to(device)
    autoencoder.load_state_dict(torch.load(f'ckpt/autoencoder_{category}_epoch_{90}.pt'))
    autoencoder.eval()
    
    print(f"[*] Training Diffusion Model for {category}...")
    diffusion = ImprovedLatentDiffusionModel(autoencoder).to(device)
    train_diffusion(diffusion, train_loader, val_loader, category, device)

    # USE CHECKPOINT
    # diffusion = ImprovedLatentDiffusionModel(autoencoder).to(device)
    # diffusion.load_state_dict(torch.load(f'ckpt/diffusion_{category}_epoch_{diff_epoch}.pt'))
    # diffusion.eval()
    
    print("[*] Generating samples...")
    sample_path = generate_evaluation_samples(diffusion, category)
    
    print("[*] Running evaluation...")
    os.system(f'python eval.py {category} {sample_path}')
    
    wandb.finish()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('ckpt', exist_ok=True)
    os.makedirs('generated_samples', exist_ok=True)
    train_and_evaluate('chair', device=device)

if __name__ == '__main__':
    main()

