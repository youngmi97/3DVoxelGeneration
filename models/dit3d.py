import torch
import torch.nn as nn
import numpy as np
import math
from .utils_vit import Attention, Mlp, window_partition, window_unpartition

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
    def __init__(
        self, 
        hidden_size, 
        num_heads, 
        mlp_ratio=4.0,
        window_size=0,
        input_size=None,
        attn_drop=0.,
        proj_drop=0.,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            eta=None
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=proj_drop,
            eta=None
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.window_size = window_size
        self.input_size = input_size

    def forward(self, x, c):
        B = x.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        shortcut = x
        x = self.norm1(x)
        
        # Apply window partition if needed
        if self.window_size > 0:
            x = x.view(B, self.input_size[0], self.input_size[1], self.input_size[2], -1)
            x, pad_xyz = window_partition(x, self.window_size)
            win_size_cubed = self.window_size ** 3
            x = x.view(-1, win_size_cubed, x.shape[-1])
            
            # Adjust modulation parameters for window attention
            num_windows = x.shape[0] // B
            shift_msa = shift_msa.repeat_interleave(num_windows, dim=0)
            scale_msa = scale_msa.repeat_interleave(num_windows, dim=0)
            
        # Modulate and apply attention
        x = modulate(x, shift_msa, scale_msa)
        x = self.attn(x)

        # Reverse window partition if needed
        if self.window_size > 0:
            x = x.view(-1, self.window_size, self.window_size, self.window_size, x.shape[-1])
            x = window_unpartition(x, self.window_size, pad_xyz, 
                                 (self.input_size[0], self.input_size[1], self.input_size[2]))
            x = x.view(B, -1, x.shape[-1])

        x = shortcut + gate_msa.unsqueeze(1) * x
        
        # MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class DiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=4,
        in_channels=1,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=False,
        window_size=0,
        window_block_indexes=(),
        attn_drop=0.,
        proj_drop=0.,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.input_size = input_size

        # Patch embedding
        self.patch_embed = nn.Conv3d(in_channels, hidden_size, 
                                   kernel_size=patch_size, stride=patch_size)
        
        # Calculate patches info
        self.n_patches = (input_size // patch_size) ** 3
        self.patch_dim = (input_size // patch_size, input_size // patch_size, input_size // patch_size)
        
        # Embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=self.patch_dim,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            ) for i in range(depth)
        ])

        # Final layers
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.output = nn.Linear(hidden_size, patch_size**3 * self.out_channels, bias=True)
        
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize patch_embed
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize position embedding
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], 
                                          self.patch_dim[0])
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize other weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.output.weight, 0)
        nn.init.constant_(self.output.bias, 0)

    def unpatchify(self, x):
        """Convert patched representation back to voxel grid."""
        p = self.patch_size
        c = self.out_channels
        h = w = d = self.patch_dim[0]
        
        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, c))
        x = torch.einsum('nhwdpqrc->nchpwqdr', x)
        x = x.reshape(shape=(x.shape[0], c, h * p, w * p, d * p))
        return x

    def forward(self, x, t):
        """
        Forward pass of DiT.
        x: (B, C, H, W, D) tensor of spatial inputs
        t: (B,) tensor of diffusion timesteps
        """
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        
        # Time embedding
        t = self.t_embedder(t)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t)
            
        # Output
        x = self.norm_final(x)
        x = self.output(x)
        x = self.unpatchify(x)
        # print(f"DiT output - mean: {x.mean():.4f}, std: {x.std():.4f}")
        
        return x

def get_3d_sincos_pos_embed(embed_dim, grid_size):
    """Generate sinusoidal positional embeddings."""
    grid_x = np.arange(grid_size, dtype=np.float32)
    grid_y = np.arange(grid_size, dtype=np.float32)
    grid_z = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    
    embed_dim_third = embed_dim // 3
    pos_embed = np.concatenate([
        get_1d_sincos_pos_embed_from_grid(embed_dim_third, grid[0]),
        get_1d_sincos_pos_embed_from_grid(embed_dim_third, grid[1]),
        get_1d_sincos_pos_embed_from_grid(embed_dim_third, grid[2])
    ], axis=1)
    
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate sinusoidal position embedding for a grid."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def DiT_S(input_size=32, **kwargs):
    return DiT(
        input_size=input_size,
        hidden_size=384,
        depth=12,
        num_heads=6,
        patch_size=4,
        window_size=4,
        window_block_indexes=tuple(range(0, 12, 2)),
        attn_drop=0.1,
        proj_drop=0.1,
        **kwargs
    )
    
def DiT_L_4(input_size=64, pretrained=False, **kwargs):
    return DiT(
        input_size=input_size,
        hidden_size=1152,
        depth=24,
        num_heads=16,
        patch_size=4,
        window_size=4,
        window_block_indexes=tuple(range(0, 12, 2)),
        attn_drop=0.1,
        proj_drop=0.1,
        **kwargs
    )
    
def DiT_M_4(input_size=64, pretrained=False, **kwargs):
    return DiT(
        input_size=input_size,
        hidden_size=1152,
        depth=16,
        num_heads=16,
        patch_size=4,
        window_size=4,
        window_block_indexes=tuple(range(0, 12, 2)),
        attn_drop=0.1,
        proj_drop=0.1,
        **kwargs
    )