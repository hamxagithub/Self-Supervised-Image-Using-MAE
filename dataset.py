"""
TinyImageNet Dataset Loader for MAE Training
Designed for Kaggle environment
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob


class TinyImageNetDataset(Dataset):
    """
    TinyImageNet Dataset for MAE training.
    Images are 64x64 originally, we resize to 224x224.
    """
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Root directory of TinyImageNet dataset
            split: 'train' or 'val'
            transform: Optional torchvision transforms
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.image_paths = []
        
        if split == 'train':
            # Training images are in train/*/images/*.JPEG
            train_dir = os.path.join(root_dir, 'train')
            if os.path.exists(train_dir):
                for class_dir in os.listdir(train_dir):
                    class_images_dir = os.path.join(train_dir, class_dir, 'images')
                    if os.path.isdir(class_images_dir):
                        for img_name in os.listdir(class_images_dir):
                            if img_name.endswith('.JPEG') or img_name.endswith('.jpeg') or img_name.endswith('.jpg'):
                                self.image_paths.append(os.path.join(class_images_dir, img_name))
        else:
            # Validation images
            val_dir = os.path.join(root_dir, 'val', 'images')
            if os.path.exists(val_dir):
                for img_name in os.listdir(val_dir):
                    if img_name.endswith('.JPEG') or img_name.endswith('.jpeg') or img_name.endswith('.jpg'):
                        self.image_paths.append(os.path.join(val_dir, img_name))
                        
        print(f"Found {len(self.image_paths)} images in {split} split")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image


def get_mae_transforms(img_size=224, is_train=True):
    """
    Get transforms for MAE training/validation.
    """
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return transform


def get_inverse_transform():
    """
    Get inverse transform to convert normalized tensor back to image.
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    return inv_normalize


def denormalize(tensor):
    """
    Denormalize a tensor for visualization.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    if tensor.device != mean.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor


def create_dataloaders(
    data_dir,
    batch_size=32,
    img_size=224,
    num_workers=4
):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Path to TinyImageNet dataset
        batch_size: Batch size
        img_size: Image size (default 224)
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader
    """
    train_transform = get_mae_transforms(img_size, is_train=True)
    val_transform = get_mae_transforms(img_size, is_train=False)
    
    train_dataset = TinyImageNetDataset(
        root_dir=data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = TinyImageNetDataset(
        root_dir=data_dir,
        split='val',
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading (adjust path for local testing)
    data_dir = "/kaggle/input/tiny-imagenet/tiny-imagenet-200"
    
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=32,
        img_size=224,
        num_workers=2
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test one batch
    for batch in train_loader:
        print(f"Batch shape: {batch.shape}")
        break
