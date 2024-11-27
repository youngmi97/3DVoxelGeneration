import torch
from tqdm import tqdm

class VoxelDiffusion:
    def __init__(self, eps_model, n_steps=1000, device="cuda"):
        self.eps_model = eps_model
        self.n_steps = n_steps
        self.device = device
        
        # Define beta schedule
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion process for binary voxels"""
        if noise is None:
            noise = torch.randn_like(x_0)
            
        alpha_t = self.alpha_bar[t]
        # Ensure x_0 is binary
        x_0 = (x_0 > 0.5).float()
        return (
            torch.sqrt(alpha_t)[:, None, None, None, None] * x_0 +
            torch.sqrt(1 - alpha_t)[:, None, None, None, None] * noise
        )
    
    def p_sample(self, x_t, t):
        eps_theta = self.eps_model(x_t, t)
        alpha_t = self.alpha_bar[t]
        alpha_t_prev = torch.where(t > 0, self.alpha_bar[t - 1], torch.ones_like(t, device=self.device))
        
        sigma_t = torch.sqrt(self.beta[t])
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        pred_x0 = (x_t - sqrt_one_minus_alpha_t[:, None, None, None, None] * eps_theta) / \
                (torch.sqrt(alpha_t)[:, None, None, None, None] + eps)
        
        dir_xt = torch.sqrt((1 - alpha_t_prev - sigma_t**2).clamp(min=eps))[:, None, None, None, None] * eps_theta
        mean = torch.sqrt(alpha_t_prev)[:, None, None, None, None] * pred_x0 + dir_xt
        
        noise = torch.randn_like(x_t) if t[0] > 0 else 0
        x_next = mean + sigma_t[:, None, None, None, None] * noise
        
        if t[0] == 0:
            x_next = (x_next > 0.5).float()
        
        return x_next
    
    def generate_samples(self, n_samples, shape=(32, 32, 32)):
        """Generate binary voxel samples"""
        self.eps_model.eval()
        with torch.no_grad():
            x_t = torch.randn(n_samples, 1, *shape).to(self.device)
            
            # Add print statements
            print(f"Initial noise mean: {x_t.mean().item():.4f}, std: {x_t.std().item():.4f}")
            
            for t in tqdm(reversed(range(self.n_steps)), desc="Sampling"):
                t_tensor = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
                x_t = self.p_sample(x_t, t_tensor)
                
                # Print intermediate values occasionally
                if t % 100 == 0:
                    print(f"Step {t} - mean: {x_t.mean().item():.4f}, std: {x_t.std().item():.4f}")
                    print(f"Number of non-zero: {(x_t > 0.5).sum().item()}")
            
            # Ensure final output is binary with explicit threshold
            samples = (x_t > 0.5).float()
            print(f"Final non-zero count: {samples.sum().item()}")
            
        return samples