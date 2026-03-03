"""
Evaluation Metrics for MAE
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- MSE (Mean Squared Error)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import math


def denormalize_for_metrics(tensor):
    """
    Denormalize tensor from ImageNet normalization for metric calculation.
    Returns tensor in range [0, 1].
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def calculate_psnr(pred, target, max_val=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        pred: Predicted image tensor (B, C, H, W) or (C, H, W)
        target: Target image tensor (same shape as pred)
        max_val: Maximum pixel value (1.0 for normalized images)
    
    Returns:
        PSNR value in dB
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    mse = F.mse_loss(pred, target, reduction='mean')
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * math.log10(max_val) - 10 * torch.log10(mse)
    return psnr.item()


def calculate_psnr_batch(pred, target, max_val=1.0):
    """
    Calculate PSNR for each image in batch.
    
    Args:
        pred: Predicted images (B, C, H, W)
        target: Target images (B, C, H, W)
        max_val: Maximum pixel value
    
    Returns:
        List of PSNR values for each image
    """
    batch_size = pred.shape[0]
    psnr_values = []
    
    for i in range(batch_size):
        psnr = calculate_psnr(pred[i], target[i], max_val)
        psnr_values.append(psnr)
    
    return psnr_values


def gaussian_kernel(size=11, sigma=1.5, channels=3, device='cpu'):
    """
    Create Gaussian kernel for SSIM calculation.
    """
    # Create 1D Gaussian kernel
    x = torch.arange(size, device=device).float() - size // 2
    gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    
    # Create 2D Gaussian kernel
    gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
    
    # Expand for convolution
    kernel = gauss_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    
    return kernel


def calculate_ssim(pred, target, window_size=11, sigma=1.5, data_range=1.0):
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        pred: Predicted image tensor (B, C, H, W) or (C, H, W)
        target: Target image tensor (same shape as pred)
        window_size: Size of Gaussian kernel
        sigma: Standard deviation of Gaussian kernel
        data_range: Range of pixel values
    
    Returns:
        SSIM value (higher is better, max 1.0)
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    device = pred.device
    channels = pred.shape[1]
    
    # Constants for numerical stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Create Gaussian kernel
    kernel = gaussian_kernel(window_size, sigma, channels, device)
    
    # Compute means
    mu_pred = F.conv2d(pred, kernel, padding=window_size // 2, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=window_size // 2, groups=channels)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    # Compute variances and covariance
    sigma_pred_sq = F.conv2d(pred ** 2, kernel, padding=window_size // 2, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, kernel, padding=window_size // 2, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=channels) - mu_pred_target
    
    # SSIM formula
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    
    ssim_map = numerator / denominator
    
    # Return mean SSIM
    return ssim_map.mean().item()


def calculate_ssim_batch(pred, target, window_size=11, sigma=1.5, data_range=1.0):
    """
    Calculate SSIM for each image in batch.
    
    Args:
        pred: Predicted images (B, C, H, W)
        target: Target images (B, C, H, W)
    
    Returns:
        List of SSIM values for each image
    """
    batch_size = pred.shape[0]
    ssim_values = []
    
    for i in range(batch_size):
        ssim = calculate_ssim(pred[i:i+1], target[i:i+1], window_size, sigma, data_range)
        ssim_values.append(ssim)
    
    return ssim_values


def evaluate_mae_model(
    model,
    dataloader,
    mask_ratio=0.75,
    num_batches=None,
    device='cuda'
):
    """
    Comprehensive evaluation of MAE model.
    
    Args:
        model: Trained MAE model
        dataloader: DataLoader with images
        mask_ratio: Ratio of patches to mask
        num_batches: Number of batches to evaluate (None for all)
        device: Device to run on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_psnr = []
    all_ssim = []
    all_mse = []
    
    if isinstance(model, nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
    
    total_batches = len(dataloader) if num_batches is None else min(num_batches, len(dataloader))
    
    with torch.no_grad():
        for batch_idx, images in enumerate(tqdm(dataloader, desc="Evaluating", total=total_batches)):
            if num_batches is not None and batch_idx >= num_batches:
                break
            
            images = images.to(device)
            
            # Get predictions
            pred, target, mask_indices = model_module(images, mask_ratio)
            
            # Convert patches to images
            pred_images = model_module.unpatchify(pred)
            target_images = model_module.unpatchify(target)
            
            # Denormalize for metrics
            pred_images = denormalize_for_metrics(pred_images)
            target_images = denormalize_for_metrics(target_images)
            
            # Calculate metrics
            batch_psnr = calculate_psnr_batch(pred_images, target_images)
            batch_ssim = calculate_ssim_batch(pred_images, target_images)
            batch_mse = F.mse_loss(pred_images, target_images, reduction='none').mean(dim=[1,2,3])
            
            all_psnr.extend(batch_psnr)
            all_ssim.extend(batch_ssim)
            all_mse.extend(batch_mse.cpu().tolist())
    
    # Compute statistics
    results = {
        'psnr': {
            'mean': np.mean(all_psnr),
            'std': np.std(all_psnr),
            'min': np.min(all_psnr),
            'max': np.max(all_psnr)
        },
        'ssim': {
            'mean': np.mean(all_ssim),
            'std': np.std(all_ssim),
            'min': np.min(all_ssim),
            'max': np.max(all_ssim)
        },
        'mse': {
            'mean': np.mean(all_mse),
            'std': np.std(all_mse),
            'min': np.min(all_mse),
            'max': np.max(all_mse)
        },
        'num_samples': len(all_psnr)
    }
    
    return results


def evaluate_masked_regions_only(
    model,
    dataloader,
    mask_ratio=0.75,
    num_batches=None,
    device='cuda'
):
    """
    Evaluate reconstruction quality only on masked regions.
    
    Args:
        model: Trained MAE model
        dataloader: DataLoader with images
        mask_ratio: Ratio of patches to mask
        num_batches: Number of batches to evaluate
        device: Device to run on
    
    Returns:
        Dictionary with evaluation metrics for masked regions
    """
    model.eval()
    
    all_psnr_masked = []
    all_ssim_masked = []
    
    if isinstance(model, nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
    
    total_batches = len(dataloader) if num_batches is None else min(num_batches, len(dataloader))
    
    with torch.no_grad():
        for batch_idx, images in enumerate(tqdm(dataloader, desc="Evaluating masked regions", total=total_batches)):
            if num_batches is not None and batch_idx >= num_batches:
                break
            
            images = images.to(device)
            B = images.shape[0]
            
            # Get predictions
            pred, target, mask_indices = model_module(images, mask_ratio)
            
            # Get only masked patches
            mask_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, pred.shape[-1])
            pred_masked = torch.gather(pred, dim=1, index=mask_indices_expanded)
            target_masked = torch.gather(target, dim=1, index=mask_indices_expanded)
            
            # Reshape for metric calculation (treat patches as small images)
            patch_size = model_module.patch_size
            num_masked = pred_masked.shape[1]
            
            pred_patches = pred_masked.reshape(B * num_masked, 3, patch_size, patch_size)
            target_patches = target_masked.reshape(B * num_masked, 3, patch_size, patch_size)
            
            # Denormalize
            pred_patches = denormalize_for_metrics_patches(pred_patches, patch_size)
            target_patches = denormalize_for_metrics_patches(target_patches, patch_size)
            
            # Calculate PSNR for patches
            for i in range(pred_patches.shape[0]):
                psnr = calculate_psnr(pred_patches[i], target_patches[i])
                all_psnr_masked.append(psnr)
    
    results = {
        'psnr_masked': {
            'mean': np.mean(all_psnr_masked),
            'std': np.std(all_psnr_masked),
            'min': np.min(all_psnr_masked),
            'max': np.max(all_psnr_masked)
        },
        'num_patches': len(all_psnr_masked)
    }
    
    return results


def denormalize_for_metrics_patches(patches, patch_size):
    """
    Denormalize patches for metrics.
    patches: (N, 3*patch_size*patch_size)
    """
    if patches.dim() == 2:
        # Reshape from flat patches to image format
        B = patches.shape[0]
        patches = patches.reshape(B, 3, patch_size, patch_size)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(patches.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(patches.device)
    
    patches = patches * std + mean
    patches = torch.clamp(patches, 0, 1)
    return patches


def print_evaluation_results(results):
    """
    Pretty print evaluation results.
    """
    print("\n" + "="*50)
    print("MAE Evaluation Results")
    print("="*50)
    
    print(f"\nNumber of samples evaluated: {results['num_samples']}")
    
    print("\nPSNR (Peak Signal-to-Noise Ratio):")
    print(f"  Mean: {results['psnr']['mean']:.2f} dB")
    print(f"  Std:  {results['psnr']['std']:.2f} dB")
    print(f"  Min:  {results['psnr']['min']:.2f} dB")
    print(f"  Max:  {results['psnr']['max']:.2f} dB")
    
    print("\nSSIM (Structural Similarity Index):")
    print(f"  Mean: {results['ssim']['mean']:.4f}")
    print(f"  Std:  {results['ssim']['std']:.4f}")
    print(f"  Min:  {results['ssim']['min']:.4f}")
    print(f"  Max:  {results['ssim']['max']:.4f}")
    
    print("\nMSE (Mean Squared Error):")
    print(f"  Mean: {results['mse']['mean']:.6f}")
    print(f"  Std:  {results['mse']['std']:.6f}")
    print(f"  Min:  {results['mse']['min']:.6f}")
    print(f"  Max:  {results['mse']['max']:.6f}")
    
    print("="*50 + "\n")


if __name__ == "__main__":
    # Test metrics
    print("Testing PSNR and SSIM metrics...")
    
    # Create test tensors
    img1 = torch.randn(2, 3, 224, 224)
    img2 = img1 + torch.randn_like(img1) * 0.1  # Add noise
    
    # Normalize to valid range
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # Test PSNR
    psnr = calculate_psnr(img1[0], img2[0])
    print(f"PSNR: {psnr:.2f} dB")
    
    # Test SSIM
    ssim = calculate_ssim(img1, img2)
    print(f"SSIM: {ssim:.4f}")
    
    print("\nMetrics module loaded successfully!")
