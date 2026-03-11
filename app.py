"""
🎭 Masked Autoencoder (MAE) - HuggingFace Spaces App
Beautiful UI for image reconstruction with detailed metrics
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from einops import rearrange
import math


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class PatchEmbed(nn.Module):
    """Convert image to patches and embed them."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Feedforward network."""
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Generate 2D sinusoidal positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
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


class ViTEncoder(nn.Module):
    """Vision Transformer Encoder (ViT-Base)."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()
        
    def _init_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
    
    def forward(self, x, visible_indices):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = torch.gather(x, dim=1, index=visible_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


class ViTDecoder(nn.Module):
    """Vision Transformer Decoder (ViT-Small)."""
    def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=12, 
                 num_heads=6, mlp_ratio=4.0, encoder_embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.pred = nn.Linear(embed_dim, patch_size ** 2 * 3)
        self._init_weights()
        
    def _init_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.normal_(self.mask_token, std=0.02)
    
    def forward(self, x, visible_indices, mask_indices):
        B, num_visible, _ = x.shape
        num_masked = mask_indices.shape[1]
        x = self.encoder_to_decoder(x)
        mask_tokens = self.mask_token.expand(B, num_masked, -1).to(dtype=x.dtype)
        full_tokens = torch.zeros(B, self.num_patches, x.shape[-1], device=x.device, dtype=x.dtype)
        visible_indices_expanded = visible_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        full_tokens.scatter_(1, visible_indices_expanded, x)
        mask_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        full_tokens.scatter_(1, mask_indices_expanded, mask_tokens)
        full_tokens = full_tokens + self.pos_embed.to(dtype=x.dtype)
        for block in self.blocks:
            full_tokens = block(full_tokens)
        full_tokens = self.norm(full_tokens)
        pred = self.pred(full_tokens)
        return pred


class MaskedAutoencoder(nn.Module):
    """Masked Autoencoder for Self-Supervised Learning."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_embed_dim=384, decoder_depth=12, decoder_num_heads=6,
                 mlp_ratio=4.0, mask_ratio=0.75):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        
        self.encoder = ViTEncoder(img_size, patch_size, in_channels, encoder_embed_dim, 
                                   encoder_depth, encoder_num_heads, mlp_ratio)
        self.decoder = ViTDecoder(img_size, patch_size, decoder_embed_dim, decoder_depth, 
                                   decoder_num_heads, mlp_ratio, encoder_embed_dim)
        
    def patchify(self, imgs):
        p = self.patch_size
        x = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        return x
    
    def unpatchify(self, x):
        p = self.patch_size
        h = w = self.img_size // p
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=p, p2=p, c=3)
        return x
    
    def random_masking(self, batch_size, device, mask_ratio=None):
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        num_patches = self.num_patches
        num_visible = int(num_patches * (1 - mask_ratio))
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        visible_indices = torch.sort(ids_shuffle[:, :num_visible], dim=1)[0]
        mask_indices = torch.sort(ids_shuffle[:, num_visible:], dim=1)[0]
        return visible_indices, mask_indices
    
    def forward(self, imgs, mask_ratio=None):
        B = imgs.shape[0]
        device = imgs.device
        visible_indices, mask_indices = self.random_masking(B, device, mask_ratio)
        latent = self.encoder(imgs, visible_indices)
        pred = self.decoder(latent, visible_indices, mask_indices)
        target = self.patchify(imgs)
        return pred, target, mask_indices
    
    def forward_loss(self, imgs, mask_ratio=None):
        pred, target, mask_indices = self.forward(imgs, mask_ratio)
        B = imgs.shape[0]
        mask_indices_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, pred.shape[-1])
        pred_masked = torch.gather(pred, dim=1, index=mask_indices_expanded)
        target_masked = torch.gather(target, dim=1, index=mask_indices_expanded)
        loss = F.mse_loss(pred_masked, target_masked)
        return loss, pred, target, mask_indices


# ============================================================================
# METRICS
# ============================================================================

def gaussian_kernel(size=11, sigma=1.5, channels=3, device='cpu'):
    """Create Gaussian kernel for SSIM calculation."""
    x = torch.arange(size, device=device).float() - size // 2
    gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
    kernel = gauss_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return kernel


def calculate_psnr(pred, target, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = F.mse_loss(pred, target, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(max_val) - 10 * torch.log10(mse)
    return psnr.item()


def calculate_ssim(pred, target, window_size=11, sigma=1.5, data_range=1.0):
    """Calculate Structural Similarity Index."""
    device = pred.device
    channels = pred.shape[1]
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    kernel = gaussian_kernel(window_size, sigma, channels, device)
    
    mu_pred = F.conv2d(pred, kernel, padding=window_size // 2, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=window_size // 2, groups=channels)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = F.conv2d(pred ** 2, kernel, padding=window_size // 2, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, kernel, padding=window_size // 2, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=channels) - mu_pred_target
    
    numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    
    ssim_map = numerator / denominator
    return ssim_map.mean().item()


def denormalize_for_metrics(tensor):
    """Denormalize tensor for metric calculation."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)


# ============================================================================
# LOAD MODEL
# ============================================================================

print("🚀 Loading MAE model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Device: {device}")

checkpoint = torch.load('mae_model_weights.pth', map_location=device)
config = checkpoint['config']

model = MaskedAutoencoder(
    img_size=config['img_size'],
    patch_size=config['patch_size'],
    encoder_embed_dim=config['encoder_embed_dim'],
    encoder_depth=config['encoder_depth'],
    encoder_num_heads=config['encoder_num_heads'],
    decoder_embed_dim=config['decoder_embed_dim'],
    decoder_depth=config['decoder_depth'],
    decoder_num_heads=config['decoder_num_heads'],
    mask_ratio=config['mask_ratio']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print("✅ Model loaded successfully!")


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return torch.clamp(tensor * std + mean, 0, 1)


def create_masked_vis(image_tensor, mask_indices, patch_size=16):
    """Create visualization of masked image with gray patches."""
    img = denormalize(image_tensor.clone())
    num_patches_per_side = 224 // patch_size
    for idx in mask_indices:
        row = idx.item() // num_patches_per_side
        col = idx.item() % num_patches_per_side
        img[:, row * patch_size:(row + 1) * patch_size,
            col * patch_size:(col + 1) * patch_size] = 0.5
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

@torch.no_grad()
def reconstruct_image(input_image, mask_ratio_percent):
    if input_image is None:
        return None, None, None, "⚠️ Please upload an image first."
    
    # Convert percentage to ratio
    mask_ratio = mask_ratio_percent / 100.0
    mask_ratio = max(0.01, min(0.99, mask_ratio))  # Clamp between 1% and 99%
    
    # Convert to PIL if needed
    if isinstance(input_image, np.ndarray):
        input_image = Image.fromarray(input_image)
    if input_image.mode != 'RGB':
        input_image = input_image.convert('RGB')
    
    # Process image
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # Forward pass with loss
    loss, pred, target, mask_indices = model.forward_loss(input_tensor, mask_ratio)
    reconstructed = model.unpatchify(pred)
    
    # Original image
    original_img = denormalize(input_tensor[0].cpu())
    original_img = (original_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Masked image
    masked_img = create_masked_vis(input_tensor[0].cpu(), mask_indices[0].cpu())
    
    # Reconstructed image
    recon_img = denormalize(reconstructed[0].cpu())
    recon_img = (recon_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Calculate metrics
    pred_denorm = denormalize_for_metrics(reconstructed)
    target_denorm = denormalize_for_metrics(input_tensor)
    psnr = calculate_psnr(pred_denorm, target_denorm)
    ssim = calculate_ssim(pred_denorm, target_denorm)
    
    # Determine quality rating
    if psnr >= 30 and ssim >= 0.85:
        quality = "🎯 Excellent"
        quality_color = "#10b981"
    elif psnr >= 25 and ssim >= 0.75:
        quality = "✅ Good"
        quality_color = "#3b82f6"
    elif psnr >= 20 and ssim >= 0.65:
        quality = "⚡ Fair"
        quality_color = "#f59e0b"
    else:
        quality = "🔧 Needs Improvement"
        quality_color = "#ef4444"
    
    # Create detailed metrics text
    metrics_text = f"""
## 📊 Reconstruction Quality: <span style="color: {quality_color}; font-weight: bold;">{quality}</span>

### 🎯 Detailed Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **MSE Loss** | `{loss.item():.6f}` | Mean Squared Error (Lower is better) |
| **PSNR** | `{psnr:.2f} dB` | Peak Signal-to-Noise Ratio (Higher is better) |
| **SSIM** | `{ssim:.4f}` | Structural Similarity (Closer to 1 is better) |

### 🎭 Masking Configuration

| Parameter | Value |
|-----------|-------|
| **Masking Ratio** | {mask_ratio*100:.1f}% |
| **Masked Patches** | {mask_indices.shape[1]} / 196 patches |
| **Visible Patches** | {196 - mask_indices.shape[1]} / 196 patches |
| **Patch Size** | 16×16 pixels |

### 🏗️ Model Architecture

- **Encoder**: ViT-Base (768d, 12 layers, 12 heads) ~ 86M parameters
- **Decoder**: ViT-Small (384d, 12 layers, 6 heads) ~ 22M parameters
- **Total Parameters**: ~108M
- **Training Dataset**: TinyImageNet

### 💡 Quality Guidelines

- **Excellent** (PSNR ≥ 30 dB, SSIM ≥ 0.85): Near-perfect reconstruction
- **Good** (PSNR ≥ 25 dB, SSIM ≥ 0.75): High-quality reconstruction
- **Fair** (PSNR ≥ 20 dB, SSIM ≥ 0.65): Acceptable reconstruction
- **Needs Improvement** (Below thresholds): Challenging conditions

---

💡 **Tip**: Lower masking ratios (10-50%) produce better reconstructions. Higher ratios (70-95%) test the model's limits!
"""
    
    return original_img, masked_img, recon_img, metrics_text


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS for beautiful UI
custom_css = """
#title {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    font-weight: bold;
    margin-bottom: 0.5em;
}

#subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1.2em;
    margin-bottom: 2em;
}

.gradio-container {
    max-width: 1400px;
    margin: auto;
}

#image-output img {
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

#metrics-box {
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    border-radius: 12px;
    padding: 20px;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="MAE Image Reconstruction") as demo:
    gr.HTML("""
        <h1 id="title">🎭 Masked Autoencoder (MAE)</h1>
        <p id="subtitle">Self-Supervised Image Reconstruction with Vision Transformers</p>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload & Configure")
            input_image = gr.Image(label="Upload Image", type="pil", height=300)
            
            mask_ratio_slider = gr.Slider(
                minimum=1, 
                maximum=99, 
                value=75, 
                step=1,
                label="🎭 Masking Ratio (%)", 
                info="Percentage of image patches to hide (1% = easy, 99% = extremely hard)"
            )
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                reconstruct_btn = gr.Button("🔄 Reconstruct", variant="primary", size="lg")
            
            gr.Markdown("""
            ### ℹ️ How It Works
            
            1. **Upload** any image
            2. **Adjust** the masking ratio
            3. **Click** Reconstruct
            4. **View** the results & metrics
            
            The model randomly masks patches of your image and reconstructs the full image from only the visible parts!
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 🖼️ Reconstruction Results")
            with gr.Row():
                original_output = gr.Image(label="📷 Original (224×224)", elem_id="image-output")
                masked_output = gr.Image(label="🎭 Masked Input", elem_id="image-output")
                reconstructed_output = gr.Image(label="✨ Reconstruction", elem_id="image-output")
            
            gr.Markdown("### 📊 Quality Metrics & Analysis")
            metrics_output = gr.Markdown(value="Upload an image and click **Reconstruct** to see detailed metrics.", elem_id="metrics-box")
    
    gr.Markdown("""
    ---
    ### 🎯 Try These Examples:
    
    - **Easy (10-30% masking)**: Clear reconstruction, tests basic capability
    - **Medium (40-60% masking)**: Balanced challenge, realistic scenarios
    - **Hard (70-85% masking)**: Significant challenge, impressive results
    - **Extreme (90-99% masking)**: Model's absolute limits
    
    ### 🔬 About MAE
    
    Masked Autoencoders (MAE) are self-supervised learning models that learn visual representations by reconstructing masked images. This implementation uses:
    - **Asymmetric Encoder-Decoder**: Efficient processing of visible patches
    - **ViT Architecture**: Transformer-based vision understanding
    - **High Masking Ratio**: Learns robust features from limited information
    
    📄 **Paper**: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (He et al., 2021)
    """)
    
    # Event handlers
    reconstruct_btn.click(
        fn=reconstruct_image,
        inputs=[input_image, mask_ratio_slider],
        outputs=[original_output, masked_output, reconstructed_output, metrics_output]
    )
    
    mask_ratio_slider.release(
        fn=reconstruct_image,
        inputs=[input_image, mask_ratio_slider],
        outputs=[original_output, masked_output, reconstructed_output, metrics_output]
    )
    
    clear_btn.click(
        fn=lambda: (None, None, None, None, "Upload an image to begin."),
        outputs=[input_image, original_output, masked_output, reconstructed_output, metrics_output]
    )


# Launch
if __name__ == "__main__":
    demo.launch(share=False)
