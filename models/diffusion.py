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
        """Single step of reverse diffusion process"""
        eps_theta = self.eps_model(x_t, t)
        alpha_t = self.alpha_bar[t]
        
        # Handle alpha_t_prev for batch of timesteps
        alpha_t_prev = torch.where(
            t > 0,
            self.alpha_bar[t - 1],
            torch.ones_like(t, device=self.device)
        )
        
        # Calculate required values
        sigma_t = torch.sqrt(self.beta[t])
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        # Calculate mean
        pred_x0 = (x_t - sqrt_one_minus_alpha_t[:, None, None, None, None] * eps_theta) / \
                  torch.sqrt(alpha_t)[:, None, None, None, None]
        dir_xt = torch.sqrt(1 - alpha_t_prev - sigma_t**2)[:, None, None, None, None] * eps_theta
        mean = torch.sqrt(alpha_t_prev)[:, None, None, None, None] * pred_x0 + dir_xt
        
        # Add noise if not final step (handle batched condition)
        noise = torch.where(
            t[:, None, None, None, None] > 0,
            torch.randn_like(x_t),
            torch.zeros_like(x_t)
        )
        x_next = mean + sigma_t[:, None, None, None, None] * noise
            
        # Ensure binary output for final step
        x_next = torch.where(
            t[:, None, None, None, None] == 0,
            (x_next > 0).float(),
            x_next
        )
        
        return x_next
    
    def generate_samples(self, n_samples, shape=(32, 32, 32)):
        """Generate binary voxel samples"""
        self.eps_model.eval()
        with torch.no_grad():
            # Start from random noise
            x_t = torch.randn(n_samples, 1, *shape).to(self.device)
            
            # Progressively denoise
            for t in tqdm(reversed(range(self.n_steps)), desc="Sampling"):
                t_tensor = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
                x_t = self.p_sample(x_t, t_tensor)
            
            # Ensure final output is binary
            samples = (x_t > 0.5).float()
            
        return samples