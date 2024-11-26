import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

def train_autoencoder(model, train_loader, val_loader, category, device='cuda', epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCELoss()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
            x = batch.float().to(device)
            optimizer.zero_grad()
            
            recon = model(x)
            loss = criterion(recon, x)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        wandb.log({'train_loss': avg_train_loss})
        
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'ckpt/autoencoder_{category}_epoch_{epoch}.pt')
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                torch.save(model.state_dict(), f'ckpt/autoencoder_{category}_best.pt')