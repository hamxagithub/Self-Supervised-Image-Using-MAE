"""
Masked Autoencoder (MAE) Model Architecture
- ViT-Base Encoder (86M parameters)
- ViT-Small Decoder (22M parameters)
Designed for Kaggle T4x2 GPUs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat


class PatchEmbedding(nn.Module):
    """Convert image to patches and embed them."""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = rearrange(x, 'b e h w -> b (h w) e')
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self Attention block."""
    
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""
    
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder (ViT-Base)
    - Patch Size: 16x16
    - Hidden Dimension: 768
    - Transformer Layers: 12
    - Attention Heads: 12
    - Parameters: ~86M
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize positional embeddings
        self._init_weights()
        
    def _init_weights(self):
        # Initialize positional embeddings with sin-cos
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
    def forward(self, x, mask_indices=None):
        """
        Forward pass for encoder.
        Args:
            x: Input image tensor (B, C, H, W)
            mask_indices: Indices of visible patches to keep (B, num_visible)
        Returns:
            Latent representations for visible tokens only
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Keep only visible patches (if masking is applied)
        if mask_indices is not None:
            B, N, D = x.shape
            # Gather visible patches
            mask_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, D)
            x = torch.gather(x, dim=1, index=mask_indices_expanded)
        
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        return x


class ViTDecoder(nn.Module):
    """
    Vision Transformer Decoder (ViT-Small)
    - Patch Size: 16x16
    - Hidden Dimension: 384
    - Transformer Layers: 12
    - Attention Heads: 6
    - Parameters: ~22M
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.0,
        encoder_embed_dim=768
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Project encoder output to decoder dimension
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, embed_dim)
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings for decoder
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Prediction head: project to pixel space
        self.pred = nn.Linear(embed_dim, patch_size ** 2 * 3)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize positional embeddings with sin-cos
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize mask token
        nn.init.normal_(self.mask_token, std=0.02)
        
    def forward(self, x, visible_indices, mask_indices):
        """
        Forward pass for decoder.
        Args:
            x: Encoder output for visible patches (B, num_visible, encoder_embed_dim)
            visible_indices: Indices of visible patches (B, num_visible)
            mask_indices: Indices of masked patches (B, num_masked)
        Returns:
            Reconstructed patches (B, num_patches, patch_size**2 * 3)
        """
        B, num_visible, _ = x.shape
        num_masked = mask_indices.shape[1]
        
        # Project encoder output to decoder dimension
        x = self.encoder_to_decoder(x)  # (B, num_visible, embed_dim)
        
        # Create mask tokens for masked positions
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        
        # Combine visible tokens and mask tokens
        full_tokens = torch.zeros(
            B, self.num_patches, x.shape[-1],
            device=x.device, dtype=x.dtype
        )
        
        # Place visible tokens
        visible_indices_expanded = visible_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        full_tokens.scatter_(1, visible_indices_expanded, x)
        
        # Place mask tokens
        mask_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        full_tokens.scatter_(1, mask_indices_expanded, mask_tokens)
        
        # Add positional embeddings
        full_tokens = full_tokens + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            full_tokens = block(full_tokens)
            
        full_tokens = self.norm(full_tokens)
        
        # Predict pixel values
        pred = self.pred(full_tokens)  # (B, num_patches, patch_size**2 * 3)
        
        return pred


class MaskedAutoencoder(nn.Module):
    """
    Masked Autoencoder for Self-Supervised Learning
    Asymmetric encoder-decoder architecture:
    - Encoder: ViT-Base (processes only visible patches)
    - Decoder: ViT-Small (reconstructs full image)
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=12,
        decoder_num_heads=6,
        mlp_ratio=4.0,
        mask_ratio=0.75,
        dropout=0.0
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        
        # Encoder
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = ViTDecoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            encoder_embed_dim=encoder_embed_dim
        )
        
    def patchify(self, imgs):
        """
        Convert images to patches.
        imgs: (B, C, H, W)
        returns: (B, num_patches, patch_size**2 * 3)
        """
        p = self.patch_size
        h = w = self.img_size // p
        x = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        return x
    
    def unpatchify(self, x):
        """
        Convert patches back to images.
        x: (B, num_patches, patch_size**2 * 3)
        returns: (B, C, H, W)
        """
        p = self.patch_size
        h = w = self.img_size // p
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=p, p2=p, c=3)
        return x
    
    def random_masking(self, batch_size, device, mask_ratio=None):
        """
        Generate random mask indices.
        Returns indices for visible and masked patches.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        num_patches = self.num_patches
        num_visible = int(num_patches * (1 - mask_ratio))
        
        # Generate random noise and sort to get random indices
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # Split into visible and masked indices
        visible_indices = ids_shuffle[:, :num_visible]
        mask_indices = ids_shuffle[:, num_visible:]
        
        # Sort visible indices to maintain positional order
        visible_indices = torch.sort(visible_indices, dim=1)[0]
        mask_indices = torch.sort(mask_indices, dim=1)[0]
        
        return visible_indices, mask_indices
    
    def forward(self, imgs, mask_ratio=None):
        """
        Forward pass.
        Args:
            imgs: Input images (B, C, H, W)
            mask_ratio: Ratio of patches to mask (default: 0.75)
        Returns:
            pred: Reconstructed patches (B, num_patches, patch_size**2 * 3)
            target: Original patches (B, num_patches, patch_size**2 * 3)
            mask_indices: Indices of masked patches
        """
        B = imgs.shape[0]
        device = imgs.device
        
        # Generate mask
        visible_indices, mask_indices = self.random_masking(B, device, mask_ratio)
        
        # Encode visible patches
        latent = self.encoder(imgs, visible_indices)
        
        # Decode to reconstruct all patches
        pred = self.decoder(latent, visible_indices, mask_indices)
        
        # Get target patches
        target = self.patchify(imgs)
        
        return pred, target, mask_indices
    
    def forward_loss(self, imgs, mask_ratio=None):
        """
        Compute forward pass and loss (MSE on masked patches only).
        """
        pred, target, mask_indices = self.forward(imgs, mask_ratio)
        
        # Compute loss only on masked patches
        B = imgs.shape[0]
        mask_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, pred.shape[-1])
        
        pred_masked = torch.gather(pred, dim=1, index=mask_indices_expanded)
        target_masked = torch.gather(target, dim=1, index=mask_indices_expanded)
        
        loss = F.mse_loss(pred_masked, target_masked)
        
        return loss, pred, target, mask_indices


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    Generate 2D sinusoidal positional embeddings.
    """
    import numpy as np
    
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Generate 2D sinusoidal positional embeddings from grid.
    """
    import numpy as np
    
    assert embed_dim % 2 == 0
    
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generate 1D sinusoidal positional embeddings.
    """
    import numpy as np
    
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def create_mae_model(
    img_size=224,
    patch_size=16,
    encoder_embed_dim=768,
    encoder_depth=12,
    encoder_num_heads=12,
    decoder_embed_dim=384,
    decoder_depth=12,
    decoder_num_heads=6,
    mask_ratio=0.75
):
    """
    Factory function to create MAE model.
    """
    model = MaskedAutoencoder(
        img_size=img_size,
        patch_size=patch_size,
        encoder_embed_dim=encoder_embed_dim,
        encoder_depth=encoder_depth,
        encoder_num_heads=encoder_num_heads,
        decoder_embed_dim=decoder_embed_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        mask_ratio=mask_ratio
    )
    return model


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    model = create_mae_model()
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Encoder parameters: {count_parameters(model.encoder):,}")
    print(f"Decoder parameters: {count_parameters(model.decoder):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    loss, pred, target, mask_indices = model.forward_loss(x)
    print(f"Loss: {loss.item():.4f}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Mask indices shape: {mask_indices.shape}")
