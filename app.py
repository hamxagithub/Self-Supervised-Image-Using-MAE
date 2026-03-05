"""
Hugging Face Spaces App for MAE Image Reconstruction
Entry point for Hugging Face deployment
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import os

# Import local modules
from mae_model import create_mae_model
from metrics import calculate_psnr, calculate_ssim, denormalize_for_metrics


class MAEInference:
    """MAE Inference wrapper for Hugging Face Spaces."""
    
    def __init__(self):
        # Use CPU for Hugging Face free tier, GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Running on: {self.device}")
        
        # Create model
        self.model = create_mae_model(
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
        
        # Load checkpoint
        self._load_weights()
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_weights(self):
        """Load model weights from various possible locations."""
        # Possible checkpoint locations
        checkpoint_paths = [
            "checkpoint_best.pth",           # Same directory (HF Spaces)
            "mae_checkpoint.pth",            # Alternative name
            "model/checkpoint_best.pth",     # Model subdirectory
            "/kaggle/working/checkpoint_best.pth",  # Kaggle
        ]
        
        for path in checkpoint_paths:
            if os.path.exists(path):
                try:
                    checkpoint = torch.load(path, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    print(f"✓ Loaded weights from: {path}")
                    return
                except Exception as e:
                    print(f"Failed to load {path}: {e}")
                    continue
        
        print("⚠ No checkpoint found - using random weights for demo")
    
    def denormalize(self, tensor):
        """Denormalize tensor for display."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor
    
    def create_masked_image(self, image_tensor, mask_indices, patch_size=16):
        """Create visualization of masked image."""
        img = self.denormalize(image_tensor.clone())
        num_patches_per_side = 224 // patch_size
        
        for idx in mask_indices:
            row = idx.item() // num_patches_per_side
            col = idx.item() % num_patches_per_side
            img[:, 
                row * patch_size:(row + 1) * patch_size,
                col * patch_size:(col + 1) * patch_size] = 0.5
        
        return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    @torch.no_grad()
    def reconstruct(self, image, mask_ratio=0.75):
        """Reconstruct image using MAE."""
        if image is None:
            return None, None, None, "Please upload an image."
        
        # Preprocess
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        pred, target, mask_indices = self.model(input_tensor, mask_ratio)
        
        # Unpatchify
        reconstructed = self.model.unpatchify(pred)
        
        # Create visualizations
        masked_img = self.create_masked_image(
            input_tensor[0].cpu(),
            mask_indices[0].cpu()
        )
        
        recon_img = self.denormalize(reconstructed[0].cpu())
        recon_img = (recon_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        original_img = self.denormalize(input_tensor[0].cpu())
        original_img = (original_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Calculate metrics
        pred_denorm = denormalize_for_metrics(reconstructed)
        target_denorm = denormalize_for_metrics(input_tensor)
        
        psnr = calculate_psnr(pred_denorm, target_denorm)
        ssim = calculate_ssim(pred_denorm, target_denorm)
        
        metrics_text = f"""
### 📊 Reconstruction Metrics
| Metric | Value |
|--------|-------|
| **PSNR** | {psnr:.2f} dB |
| **SSIM** | {ssim:.4f} |
| **Mask Ratio** | {mask_ratio*100:.0f}% |
| **Visible Patches** | {int((1-mask_ratio)*196)} / 196 |
| **Masked Patches** | {int(mask_ratio*196)} / 196 |
"""
        
        return original_img, masked_img, recon_img, metrics_text


# Initialize model globally (loaded once when app starts)
print("Initializing MAE model...")
mae = MAEInference()


def process_image(input_image, mask_ratio):
    """Main processing function for Gradio."""
    if input_image is None:
        return None, None, None, "⬆️ Please upload an image to get started."
    
    mask_ratio = max(0.1, min(0.95, mask_ratio))
    return mae.reconstruct(input_image, mask_ratio)


# Create Gradio interface
with gr.Blocks(
    title="MAE Image Reconstruction",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container { max-width: 1200px !important; }
        .output-image { border-radius: 8px; }
    """
) as demo:
    
    gr.Markdown("""
    # 🎭 Masked Autoencoder (MAE) Image Reconstruction
    
    Upload any image to see how the MAE reconstructs it from only **25% visible patches**.
    The model learns powerful visual representations by predicting masked regions.
    
    > **Try adjusting the mask ratio** to see how the reconstruction quality changes!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📤 Upload Image",
                type="pil",
                height=280
            )
            
            mask_ratio_slider = gr.Slider(
                minimum=0.1,
                maximum=0.95,
                value=0.75,
                step=0.05,
                label="🎚️ Masking Ratio",
                info="Percentage of patches to mask (default: 75%)"
            )
            
            reconstruct_btn = gr.Button(
                "🔄 Reconstruct Image",
                variant="primary",
                size="lg"
            )
            
            metrics_output = gr.Markdown(
                value="⬆️ Upload an image and click **Reconstruct** to see metrics."
            )
        
        with gr.Column(scale=2):
            with gr.Row():
                original_output = gr.Image(
                    label="Original (224×224)",
                    height=224,
                    show_download_button=True
                )
                masked_output = gr.Image(
                    label="Masked Input",
                    height=224,
                    show_download_button=True
                )
                reconstructed_output = gr.Image(
                    label="Reconstruction",
                    height=224,
                    show_download_button=True
                )
    
    # Event handlers
    reconstruct_btn.click(
        fn=process_image,
        inputs=[input_image, mask_ratio_slider],
        outputs=[original_output, masked_output, reconstructed_output, metrics_output]
    )
    
    mask_ratio_slider.change(
        fn=process_image,
        inputs=[input_image, mask_ratio_slider],
        outputs=[original_output, masked_output, reconstructed_output, metrics_output]
    )
    
    input_image.change(
        fn=process_image,
        inputs=[input_image, mask_ratio_slider],
        outputs=[original_output, masked_output, reconstructed_output, metrics_output]
    )
    
    gr.Markdown("""
    ---
    ### 🔬 How MAE Works
    
    1. **Masking**: Randomly mask ~75% of image patches
    2. **Encoding**: Process only visible patches through ViT encoder
    3. **Decoding**: Reconstruct full image using a lightweight decoder
    
    **Model Architecture:**
    - **Encoder**: ViT-Base (768 dim, 12 layers) — 86M params
    - **Decoder**: ViT-Small (384 dim, 12 layers) — 22M params
    
    📄 [Original Paper](https://arxiv.org/abs/2111.06377) | 
    🔗 [GitHub](https://github.com/facebookresearch/mae)
    """)


# Launch app
if __name__ == "__main__":
    demo.launch()
