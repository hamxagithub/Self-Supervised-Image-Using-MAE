"""
Visualization Module for MAE
- Masked input visualization
- Reconstruction visualization
- Side-by-side comparisons
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from PIL import Image
import os


def denormalize(tensor):
    """
    Denormalize a tensor for visualization.
    ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def tensor_to_numpy(tensor):
    """Convert tensor to numpy for matplotlib."""
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image from batch
    
    tensor = denormalize(tensor)
    np_img = tensor.permute(1, 2, 0).numpy()
    return np_img


def create_masked_image(image, mask_indices, patch_size=16, img_size=224):
    """
    Create visualization of masked image (visible patches only).
    
    Args:
        image: Original image tensor (C, H, W)
        mask_indices: Indices of masked patches (num_masked,)
        patch_size: Size of each patch
        img_size: Original image size
    
    Returns:
        Masked image as numpy array
    """
    # Denormalize image
    img = denormalize(image.clone())
    
    # Calculate number of patches per row/column
    num_patches_per_side = img_size // patch_size
    
    # Create mask
    masked_img = img.clone()
    
    for idx in mask_indices:
        row = idx.item() // num_patches_per_side
        col = idx.item() % num_patches_per_side
        
        # Set masked region to gray
        masked_img[:, 
                   row * patch_size:(row + 1) * patch_size,
                   col * patch_size:(col + 1) * patch_size] = 0.5
    
    return masked_img.permute(1, 2, 0).numpy()


def create_reconstruction_visualization(
    model,
    images,
    mask_ratio=0.75,
    device='cuda'
):
    """
    Create visualization of MAE reconstruction.
    
    Args:
        model: Trained MAE model
        images: Input image tensor (B, C, H, W)
        mask_ratio: Ratio of patches to mask
        device: Device to run on
    
    Returns:
        Dictionary with visualization components
    """
    model.eval()
    
    with torch.no_grad():
        images = images.to(device)
        
        # Get model predictions
        if isinstance(model, nn.DataParallel):
            pred, target, mask_indices = model.module(images, mask_ratio)
            visible_indices, _ = model.module.random_masking(
                images.shape[0], device, mask_ratio
            )
            reconstructed = model.module.unpatchify(pred)
        else:
            pred, target, mask_indices = model(images, mask_ratio)
            visible_indices, _ = model.random_masking(
                images.shape[0], device, mask_ratio
            )
            reconstructed = model.unpatchify(pred)
    
    return {
        'original': images.cpu(),
        'reconstructed': reconstructed.cpu(),
        'mask_indices': mask_indices.cpu(),
        'visible_indices': visible_indices.cpu() if 'visible_indices' in dir() else None
    }


def visualize_reconstruction(
    model,
    dataloader,
    num_samples=5,
    mask_ratio=0.75,
    save_path=None,
    device='cuda'
):
    """
    Visualize MAE reconstruction results.
    
    Args:
        model: Trained MAE model
        dataloader: DataLoader with images
        num_samples: Number of samples to visualize
        mask_ratio: Ratio of patches to mask
        save_path: Path to save visualization
        device: Device to run on
    """
    model.eval()
    
    # Get batch of images
    images = next(iter(dataloader))[:num_samples]
    
    # Get reconstructions
    results = create_reconstruction_visualization(model, images, mask_ratio, device)
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        original_np = tensor_to_numpy(results['original'][i])
        axes[i, 0].imshow(original_np)
        axes[i, 0].set_title('Original', fontsize=12)
        axes[i, 0].axis('off')
        
        # Masked image
        masked_np = create_masked_image(
            results['original'][i],
            results['mask_indices'][i]
        )
        axes[i, 1].imshow(masked_np)
        axes[i, 1].set_title(f'Masked ({int(mask_ratio*100)}%)', fontsize=12)
        axes[i, 1].axis('off')
        
        # Reconstructed image
        recon_np = tensor_to_numpy(results['reconstructed'][i])
        axes[i, 2].imshow(recon_np)
        axes[i, 2].set_title('Reconstruction', fontsize=12)
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    return fig


def visualize_attention_maps(
    model,
    image,
    layer_idx=-1,
    head_idx=0,
    save_path=None,
    device='cuda'
):
    """
    Visualize attention maps from the encoder.
    
    Args:
        model: Trained MAE model
        image: Input image tensor (1, C, H, W)
        layer_idx: Which transformer layer to visualize
        head_idx: Which attention head to visualize
        save_path: Path to save visualization
        device: Device to run on
    """
    # This is a placeholder for attention visualization
    # Would require modifying the model to return attention weights
    pass


def create_grid_visualization(
    original_images,
    masked_images,
    reconstructed_images,
    save_path=None
):
    """
    Create a grid visualization comparing original, masked, and reconstructed images.
    
    Args:
        original_images: List of original images (numpy arrays)
        masked_images: List of masked images (numpy arrays)
        reconstructed_images: List of reconstructed images (numpy arrays)
        save_path: Path to save the grid
    """
    num_images = len(original_images)
    
    fig, axes = plt.subplots(3, num_images, figsize=(3 * num_images, 9))
    
    titles = ['Original', 'Masked Input', 'Reconstruction']
    image_lists = [original_images, masked_images, reconstructed_images]
    
    for row, (title, images) in enumerate(zip(titles, image_lists)):
        for col, img in enumerate(images):
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grid visualization saved to {save_path}")
    
    plt.show()
    return fig


def visualize_patch_level_reconstruction(
    model,
    image,
    mask_ratio=0.75,
    save_path=None,
    device='cuda'
):
    """
    Visualize patch-level reconstruction quality.
    Shows per-patch MSE error as a heatmap.
    
    Args:
        model: Trained MAE model
        image: Input image tensor (1, C, H, W) or (C, H, W)
        mask_ratio: Ratio of patches to mask
        save_path: Path to save visualization
        device: Device to run on
    """
    model.eval()
    
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    with torch.no_grad():
        image = image.to(device)
        
        if isinstance(model, nn.DataParallel):
            model_module = model.module
        else:
            model_module = model
        
        pred, target, mask_indices = model_module(image, mask_ratio)
        
        # Calculate per-patch MSE
        patch_mse = ((pred - target) ** 2).mean(dim=-1)  # (B, num_patches)
        
        # Reshape to patch grid
        num_patches_per_side = int(patch_mse.shape[1] ** 0.5)
        patch_mse_grid = patch_mse[0].reshape(num_patches_per_side, num_patches_per_side)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    original_np = tensor_to_numpy(image[0].cpu())
    axes[0].imshow(original_np)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Reconstruction
    reconstructed = model_module.unpatchify(pred)
    recon_np = tensor_to_numpy(reconstructed[0].cpu())
    axes[1].imshow(recon_np)
    axes[1].set_title('Reconstruction', fontsize=12)
    axes[1].axis('off')
    
    # Error heatmap
    im = axes[2].imshow(patch_mse_grid.cpu().numpy(), cmap='hot')
    axes[2].set_title('Patch MSE Error', fontsize=12)
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Patch-level visualization saved to {save_path}")
    
    plt.show()
    return fig


def save_sample_reconstructions(
    model,
    dataloader,
    save_dir='/kaggle/working',
    num_samples=10,
    mask_ratio=0.75,
    device='cuda'
):
    """
    Save individual reconstruction samples as separate images.
    
    Args:
        model: Trained MAE model
        dataloader: DataLoader with images
        save_dir: Directory to save images
        num_samples: Number of samples to save
        mask_ratio: Ratio of patches to mask
        device: Device to run on
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    images = next(iter(dataloader))[:num_samples]
    results = create_reconstruction_visualization(model, images, mask_ratio, device)
    
    for i in range(num_samples):
        # Save original
        original_np = tensor_to_numpy(results['original'][i])
        plt.imsave(
            os.path.join(save_dir, f'sample_{i+1}_original.png'),
            original_np
        )
        
        # Save masked
        masked_np = create_masked_image(
            results['original'][i],
            results['mask_indices'][i]
        )
        plt.imsave(
            os.path.join(save_dir, f'sample_{i+1}_masked.png'),
            masked_np
        )
        
        # Save reconstruction
        recon_np = tensor_to_numpy(results['reconstructed'][i])
        plt.imsave(
            os.path.join(save_dir, f'sample_{i+1}_reconstruction.png'),
            recon_np
        )
    
    print(f"Saved {num_samples} sample reconstructions to {save_dir}")


if __name__ == "__main__":
    # Test visualization (requires trained model)
    print("Visualization module loaded successfully")
