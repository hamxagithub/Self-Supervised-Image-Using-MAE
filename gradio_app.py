"""
Gradio App for MAE Image Reconstruction
- Upload an image
- Select masking ratio
- View reconstruction in real-time
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


class MAEInferenceApp:
    """
    Inference wrapper for MAE model with Gradio interface.
    """
    
    def __init__(self, checkpoint_path=None, device='cuda'):
        """
        Initialize the inference app.
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
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
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print("Warning: No checkpoint loaded. Using random weights.")
        
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
        
    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def preprocess_image(self, image):
        """
        Preprocess input image for the model.
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)  # Add batch dimension
        
        return tensor
    
    def denormalize(self, tensor):
        """Denormalize tensor for display."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        if tensor.device.type != 'cpu':
            tensor = tensor.cpu()
        
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        return tensor
    
    def create_masked_visualization(self, image_tensor, mask_indices, patch_size=16):
        """
        Create visualization of masked image.
        
        Args:
            image_tensor: Original image tensor (C, H, W)
            mask_indices: Indices of masked patches
            patch_size: Size of each patch
        
        Returns:
            Masked image as numpy array
        """
        img = self.denormalize(image_tensor.clone())
        
        num_patches_per_side = 224 // patch_size
        
        for idx in mask_indices:
            row = idx.item() // num_patches_per_side
            col = idx.item() % num_patches_per_side
            
            # Set masked region to gray
            img[:, 
                row * patch_size:(row + 1) * patch_size,
                col * patch_size:(col + 1) * patch_size] = 0.5
        
        return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    @torch.no_grad()
    def reconstruct(self, image, mask_ratio=0.75):
        """
        Reconstruct image using MAE.
        
        Args:
            image: Input image (PIL Image or numpy array)
            mask_ratio: Ratio of patches to mask (0.0 to 0.95)
        
        Returns:
            Tuple of (masked_image, reconstructed_image, metrics_dict)
        """
        # Preprocess
        input_tensor = self.preprocess_image(image).to(self.device)
        
        # Forward pass
        pred, target, mask_indices = self.model(input_tensor, mask_ratio)
        
        # Unpatchify to get image
        reconstructed = self.model.unpatchify(pred)
        
        # Create masked visualization
        masked_img = self.create_masked_visualization(
            input_tensor[0].cpu(),
            mask_indices[0].cpu()
        )
        
        # Get reconstructed image
        recon_img = self.denormalize(reconstructed[0].cpu())
        recon_img = (recon_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Get original image for comparison
        original_img = self.denormalize(input_tensor[0].cpu())
        original_img = (original_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Calculate metrics
        pred_denorm = denormalize_for_metrics(reconstructed)
        target_denorm = denormalize_for_metrics(input_tensor)
        
        psnr = calculate_psnr(pred_denorm, target_denorm)
        ssim = calculate_ssim(pred_denorm, target_denorm)
        
        metrics = {
            'psnr': psnr,
            'ssim': ssim,
            'mask_ratio': mask_ratio,
            'visible_patches': int((1 - mask_ratio) * 196),
            'masked_patches': int(mask_ratio * 196)
        }
        
        return original_img, masked_img, recon_img, metrics


def create_gradio_interface(checkpoint_path=None):
    """
    Create Gradio interface for MAE reconstruction.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
    
    Returns:
        Gradio Blocks interface
    """
    # Initialize inference app
    app = MAEInferenceApp(checkpoint_path=checkpoint_path)
    
    def process_image(input_image, mask_ratio):
        """Process image and return results."""
        if input_image is None:
            return None, None, None, "Please upload an image."
        
        # Ensure mask_ratio is valid
        mask_ratio = max(0.1, min(0.95, mask_ratio))
        
        # Get reconstruction
        original, masked, reconstructed, metrics = app.reconstruct(
            input_image, 
            mask_ratio
        )
        
        # Format metrics
        metrics_text = f"""
### Reconstruction Metrics
- **PSNR**: {metrics['psnr']:.2f} dB
- **SSIM**: {metrics['ssim']:.4f}
- **Mask Ratio**: {metrics['mask_ratio']*100:.1f}%
- **Visible Patches**: {metrics['visible_patches']} / 196
- **Masked Patches**: {metrics['masked_patches']} / 196
"""
        
        return original, masked, reconstructed, metrics_text
    
    # Create interface
    with gr.Blocks(title="MAE Image Reconstruction", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎭 Masked Autoencoder (MAE) Image Reconstruction
        
        Upload an image and adjust the masking ratio to see how the MAE reconstructs masked regions.
        The model has learned visual representations by reconstructing images with random patches masked.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=300
                )
                
                mask_ratio_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.95,
                    value=0.75,
                    step=0.05,
                    label="Masking Ratio",
                    info="Percentage of image patches to mask (default: 75%)"
                )
                
                reconstruct_btn = gr.Button(
                    "🔄 Reconstruct",
                    variant="primary",
                    size="lg"
                )
                
                # Metrics output
                metrics_output = gr.Markdown(
                    label="Metrics",
                    value="Upload an image and click Reconstruct to see metrics."
                )
            
            with gr.Column(scale=2):
                with gr.Row():
                    original_output = gr.Image(
                        label="Original (Resized to 224x224)",
                        height=256
                    )
                    masked_output = gr.Image(
                        label="Masked Input",
                        height=256
                    )
                    reconstructed_output = gr.Image(
                        label="Reconstruction",
                        height=256
                    )
        
        # Examples
        gr.Markdown("### Try Example Images")
        gr.Examples(
            examples=[
                ["example_1.jpg", 0.75],
                ["example_2.jpg", 0.50],
                ["example_3.jpg", 0.90],
            ],
            inputs=[input_image, mask_ratio_slider],
            outputs=[original_output, masked_output, reconstructed_output, metrics_output],
            fn=process_image,
            cache_examples=False
        )
        
        # Event handlers
        reconstruct_btn.click(
            fn=process_image,
            inputs=[input_image, mask_ratio_slider],
            outputs=[original_output, masked_output, reconstructed_output, metrics_output]
        )
        
        # Auto-reconstruct when mask ratio changes
        mask_ratio_slider.change(
            fn=process_image,
            inputs=[input_image, mask_ratio_slider],
            outputs=[original_output, masked_output, reconstructed_output, metrics_output]
        )
        
        gr.Markdown("""
        ---
        ### About MAE
        
        **Masked Autoencoders (MAE)** are a self-supervised learning approach that:
        
        1. Randomly masks a large portion (75%) of image patches
        2. Processes only visible patches through an encoder
        3. Reconstructs the original image using a lightweight decoder
        
        This forces the model to learn meaningful visual representations that capture semantic content.
        
        **Architecture:**
        - **Encoder**: ViT-Base (768 dim, 12 layers, 12 heads) - ~86M parameters
        - **Decoder**: ViT-Small (384 dim, 12 layers, 6 heads) - ~22M parameters
        """)
    
    return demo


def launch_app(checkpoint_path=None, share=False, port=7860):
    """
    Launch the Gradio app.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        share: Whether to create a public link
        port: Port number for local server
    """
    demo = create_gradio_interface(checkpoint_path)
    demo.launch(share=share, server_port=port)


if __name__ == "__main__":
    # Default checkpoint path for Kaggle
    checkpoint_path = "/kaggle/working/checkpoint_best.pth"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("The app will run with random weights.")
        checkpoint_path = None
    
    # Launch app
    launch_app(checkpoint_path=checkpoint_path, share=True)
