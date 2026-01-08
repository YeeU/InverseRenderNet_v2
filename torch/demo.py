#!/usr/bin/env python3
"""
Run InverseRenderNet on a demo image.

Usage:
    python demo.py --image ../demo_im.jpg --mask ../demo_mask.jpg
"""

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from model import InverseRenderNet, load_weights


def srgb_to_linear(img):
    """Convert sRGB to linear RGB."""
    img = img.astype(np.float32) / 255.0
    mask = img <= 0.04045
    img[mask] = img[mask] / 12.92
    img[~mask] = ((img[~mask] + 0.055) / 1.055) ** 2.4
    return img


def main():
    parser = argparse.ArgumentParser(description="InverseRenderNet PyTorch Demo")
    parser.add_argument("--image", type=str, default="../demo_im.jpg", help="Input image")
    parser.add_argument("--mask", type=str, default="../demo_mask.jpg", help="Mask image")
    parser.add_argument("--weights", type=str, default="model_ckpt.pth", help="Model weights")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    args = parser.parse_args()
    
    # Create output dir
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = InverseRenderNet(n_layers=30, n_pools=4, depth_base=32)
    load_weights(model, args.weights)
    model.eval()
    
    # Load image and mask
    img = Image.open(args.image).convert('RGB')
    mask_img = Image.open(args.mask).convert('L')
    print(f"Image size: {img.size}")
    
    # Resize to ~200px on shorter side
    orig_w, orig_h = img.size
    if orig_h > orig_w:
        scale = orig_w / 200
        new_h, new_w = int(orig_h / scale), 200
    else:
        scale = orig_h / 200
        new_w, new_h = int(orig_w / scale), 200
        
    img = img.resize((new_w, new_h))
    mask_img = mask_img.resize((new_w, new_h), Image.NEAREST)
    
    # Save inputs
    img.save(output_dir / 'input.png')
    mask_img.save(output_dir / 'mask.png')
    
    # Convert to tensors
    img_np = srgb_to_linear(np.array(img))
    img_np = img_np * 2.0 - 1.0  # [-1, 1]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    
    mask_np = (np.array(mask_img) == 255).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    
    # Apply mask to input
    img_tensor = img_tensor * mask_tensor
    
    # Inference
    with torch.no_grad():
        albedo, normal, shadow = model(img_tensor, mask_tensor)
    
    # Post-process and save
    # Albedo: min-max normalize (as in original TF code)
    albedo_np = (albedo[0].permute(1, 2, 0).numpy() / 2.0 + 0.5) * mask_np[:, :, None]
    albedo_np = (albedo_np - albedo_np.min()) / (albedo_np.max() - albedo_np.min() + 1e-8)
    Image.fromarray((albedo_np * 255).astype(np.uint8)).save(output_dir / 'albedo.png')
    
    # Normal: rescale to [0, 1]
    normal_np = (normal[0].permute(1, 2, 0).numpy() + 1.0) / 2.0
    Image.fromarray((np.clip(normal_np, 0, 1) * 255).astype(np.uint8)).save(output_dir / 'normal.png')
    
    # Shadow: rescale to [0, 1]
    shadow_np = shadow[0, 0].numpy() / 2.0 + 0.5
    Image.fromarray((np.clip(shadow_np, 0, 1) * 255).astype(np.uint8)).save(output_dir / 'shadow.png')
    
    # Shading: albedo * shadow
    shading_np = albedo_np * shadow_np[:, :, None]
    Image.fromarray((np.clip(shading_np, 0, 1) * 255).astype(np.uint8)).save(output_dir / 'shading.png')
    
    print(f"Saved outputs to {output_dir}/")


if __name__ == "__main__":
    main()
