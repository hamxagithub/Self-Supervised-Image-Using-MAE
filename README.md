# 🎭 Masked Autoencoder (MAE) for Self-Supervised Learning

Self-Supervised Image Representation Learning using Masked Autoencoders (MAE) - Implementation for Kaggle T4x2 GPUs.

## 📋 Project Overview

This implementation follows an **asymmetric transformer-based encoder-decoder design**:
- **Large Vision Transformer encoder** processes only visible patches (25%)
- **Lightweight Vision Transformer decoder** reconstructs the masked content (75%)

## 🏗️ Architecture

### Encoder (ViT-Base)
| Parameter | Value |
|-----------|-------|
| Patch Size | 16 × 16 |
| Image Size | 224 × 224 |
| Hidden Dimension | 768 |
| Transformer Layers | 12 |
| Attention Heads | 12 |
| Parameters | ~86 Million |

### Decoder (ViT-Small)
| Parameter | Value |
|-----------|-------|
| Patch Size | 16 × 16 |
| Hidden Dimension | 384 |
| Transformer Layers | 12 |
| Attention Heads | 6 |
| Parameters | ~22 Million |

## 📁 Project Structure

```
MAE/
├── mae_training_notebook.ipynb   # Main Kaggle notebook (ALL-IN-ONE)
├── mae_model.py                  # MAE model architecture
├── dataset.py                    # TinyImageNet dataset loader
├── train.py                      # Training script
├── visualization.py              # Visualization utilities
├── metrics.py                    # PSNR & SSIM metrics
├── gradio_app.py                 # Gradio deployment app
└── README.md                     # This file
```

## 🚀 Kaggle Setup Instructions

### Step 1: Create New Notebook
1. Go to [Kaggle](https://www.kaggle.com/)
2. Create a new notebook
3. Set Accelerator to **GPU T4 x2**

### Step 2: Add Dataset
1. Click "Add data" in the notebook
2. Search for "tiny-imagenet" by akash2sharma
3. Add: https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet

### Step 3: Upload Notebook
1. Upload `mae_training_notebook.ipynb` to Kaggle
2. Or copy-paste the cells into a new notebook

### Step 4: Run Training
Execute all cells in order. The notebook will:
1. Install required packages (einops, gradio)
2. Define model architecture
3. Load TinyImageNet dataset
4. Train the MAE model
5. Evaluate with PSNR/SSIM metrics
6. Launch Gradio app for interactive demo

## ⚙️ Training Configuration

```python
CONFIG = {
    'img_size': 224,
    'patch_size': 16,
    'mask_ratio': 0.75,           # 75% masking
    
    # Training
    'batch_size': 64,             # Adjust if OOM
    'learning_rate': 1.5e-4,
    'weight_decay': 0.05,
    'warmup_epochs': 10,
    'total_epochs': 100,
    'gradient_clip': 1.0,
}
```

## 🔧 Key Features

- ✅ **Mixed Precision Training** (torch.cuda.amp) for memory efficiency
- ✅ **Multi-GPU Support** (DataParallel) for T4x2
- ✅ **Cosine Learning Rate Scheduler** with warmup
- ✅ **MSE Loss on masked patches only**
- ✅ **Gradient Clipping** for stability
- ✅ **Checkpointing** (best and latest models)

## 📊 Outputs

All outputs are saved to `/kaggle/working/`:

| File | Description |
|------|-------------|
| `checkpoint_best.pth` | Best model weights |
| `checkpoint_latest.pth` | Latest checkpoint |
| `training_history.json` | Loss/LR history |
| `training_curves.png` | Training plots |
| `reconstruction_samples.png` | Visual examples |
| `evaluation_results.json` | PSNR/SSIM metrics |

## 📈 Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better
- **SSIM** (Structural Similarity Index): 0-1, higher is better
- **MSE** (Mean Squared Error): Lower is better

## 🎨 Visualization

The notebook provides:
1. **Masked Input**: Shows which patches are masked (gray)
2. **Reconstruction**: Model's reconstruction
3. **Original**: Ground truth image
4. **Side-by-side comparisons** with different mask ratios

## 🌐 Gradio App

Interactive demo that allows:
- Image upload
- Adjustable masking ratio (10%-95%)
- Real-time reconstruction display
- Automatic PSNR/SSIM calculation

Launch with `share=True` to get a public Gradio link.

## 💡 Tips for Kaggle

1. **Memory Issues**: Reduce `batch_size` to 32 if OOM
2. **Quick Testing**: Set `total_epochs` to 10-20 first
3. **Resume Training**: Load from `checkpoint_latest.pth`
4. **Save Outputs**: Download `/kaggle/working/` before session ends

## 📚 References

- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (He et al., 2021)
- [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020)

## 📝 Assignment Checklist

- [x] PyTorch implementation from scratch
- [x] ViT-Base encoder (~86M params)
- [x] ViT-Small decoder (~22M params)
- [x] 75% masking with random patch selection
- [x] MSE loss on masked patches only
- [x] Mixed precision training
- [x] Cosine LR scheduler with warmup
- [x] Training loss plots
- [x] 5+ reconstruction samples
- [x] PSNR & SSIM evaluation
- [x] Gradio/Streamlit deployment app

---

**Author**: MAE Implementation for Self-Supervised Learning Assignment
