import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

def train_diffusion(model, train_loader, val_loader, category, device='cuda', epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            x = batch.float().to(device)  # [B, 1, 64, 64, 64]
            optimizer.zero_grad()
            
            # Sample random timesteps
            t = torch.rand(x.shape[0]).to(device)
            
            with autocast():
                loss = model(x, t)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        wandb.log({'diffusion_loss': avg_train_loss})
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'ckpt/diffusion_{category}_epoch_{epoch}.pt')
