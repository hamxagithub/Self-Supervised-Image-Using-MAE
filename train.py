"""
MAE Training Script
- Mixed Precision Training (AMP)
- AdamW Optimizer
- Cosine Learning Rate Scheduler
- Gradient Clipping
- Multi-GPU Support (DataParallel)
Designed for Kaggle T4x2 GPUs
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from mae_model import create_mae_model, count_parameters
from dataset import create_dataloaders, denormalize


class MAETrainer:
    """
    Trainer class for Masked Autoencoder.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        learning_rate=1.5e-4,
        weight_decay=0.05,
        warmup_epochs=10,
        total_epochs=100,
        gradient_clip=1.0,
        save_dir='/kaggle/working',
        device='cuda'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.gradient_clip = gradient_clip
        self.save_dir = save_dir
        self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Use DataParallel for multi-GPU
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Scheduler (Cosine Annealing)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-6
        )
        
        # Mixed Precision Scaler
        self.scaler = GradScaler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Best loss for checkpointing
        self.best_val_loss = float('inf')
        
    def warmup_lr(self, epoch, step, total_steps):
        """Linear warmup for learning rate."""
        if epoch < self.warmup_epochs:
            warmup_progress = (epoch * total_steps + step) / (self.warmup_epochs * total_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * warmup_progress
                
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.total_epochs}")
        
        for step, images in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            
            # Warmup learning rate
            if epoch < self.warmup_epochs:
                self.warmup_lr(epoch, step, num_batches)
            
            # Mixed precision forward pass
            with autocast():
                if isinstance(self.model, nn.DataParallel):
                    loss, _, _, _ = self.model.module.forward_loss(images)
                else:
                    loss, _, _, _ = self.model.forward_loss(images)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Step scheduler after warmup
        if epoch >= self.warmup_epochs:
            self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        for images in tqdm(self.val_loader, desc="Validation"):
            images = images.to(self.device, non_blocking=True)
            
            with autocast():
                if isinstance(self.model, nn.DataParallel):
                    loss, _, _, _ = self.model.module.forward_loss(images)
                else:
                    loss, _, _, _ = self.model.forward_loss(images)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        if isinstance(self.model, nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.save_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        return checkpoint['epoch']
    
    def train(self, start_epoch=0):
        """Full training loop."""
        print(f"Starting training from epoch {start_epoch + 1}")
        print(f"Total epochs: {self.total_epochs}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print("-" * 50)
        
        for epoch in range(start_epoch, self.total_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch+1}/{self.total_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"LR: {current_lr:.6f}, "
                  f"Time: {epoch_time:.1f}s")
            print("-" * 50)
        
        # Save training history
        self.save_history()
        self.plot_training_curves()
        
        return self.history
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curve
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Reconstruction Loss vs Epochs')
        axes[0].legend()
        axes[0].grid(True)
        
        # Learning rate curve
        axes[1].plot(self.history['learning_rate'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {fig_path}")


def train_mae(
    data_dir='/kaggle/input/tiny-imagenet/tiny-imagenet-200',
    save_dir='/kaggle/working',
    batch_size=64,
    learning_rate=1.5e-4,
    weight_decay=0.05,
    warmup_epochs=10,
    total_epochs=100,
    num_workers=4,
    resume_checkpoint=None
):
    """
    Main training function.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create model
    print("\nCreating MAE model...")
    model = create_mae_model(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=384,
        decoder_depth=12,
        decoder_num_heads=6,
        mask_ratio=0.75
    )
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Encoder parameters: {count_parameters(model.encoder):,}")
    print(f"Decoder parameters: {count_parameters(model.decoder):,}")
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=224,
        num_workers=num_workers
    )
    
    # Create trainer
    trainer = MAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        gradient_clip=1.0,
        save_dir=save_dir,
        device=device
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        start_epoch = trainer.load_checkpoint(resume_checkpoint) + 1
    
    # Train
    print("\nStarting training...")
    history = trainer.train(start_epoch=start_epoch)
    
    print("\nTraining completed!")
    return model, history


if __name__ == "__main__":
    # Training configuration for Kaggle
    model, history = train_mae(
        data_dir='/kaggle/input/tiny-imagenet/tiny-imagenet-200',
        save_dir='/kaggle/working',
        batch_size=64,  # Adjust based on GPU memory
        learning_rate=1.5e-4,
        weight_decay=0.05,
        warmup_epochs=10,
        total_epochs=100,
        num_workers=4
    )
