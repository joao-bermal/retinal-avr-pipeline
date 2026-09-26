#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for Inference and Evaluation (Study Images).
Allows taking a random image, applying full preprocessing, and extracting predictions and plots.
"""

import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config.settings import SEGMENTATION_CONFIG as C, DEVICE
from src.data.preprocessing import apply_enhanced_preprocessing
from src.models.segmentation_model import EnhancedUNet

def test_inference_on_image(image_path: Path, model_path: Path = None):
    """
    Runs inference of a trained model on an image, shows the progress of the preprocessing pipeline,
    and the generated mask.
    """
    print(f"[*] Loading image from: {image_path}")
    if not image_path.exists():
        print("❌ Image not found!")
        return

    # Original Image
    original_bgr = cv2.imread(str(image_path))
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Preprocessing
    print("[*] Applying preprocessing (CLAHE + Gamma)...")
    preprocessed_rgb = apply_enhanced_preprocessing(original_rgb)
    
    # Resize to model size
    img_size = C['DATASET']['IMAGE_SIZE']
    resized_rgb = cv2.resize(preprocessed_rgb, (img_size[1], img_size[0]))
    
    # Transform to tensor
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    normalized = (resized_rgb.astype(np.float32) / 255.0 - mean) / std
    tensor_img = torch.tensor(normalized).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    # Load Model
    model = EnhancedUNet(in_channels=3, out_channels=1, features=C['MODEL']['FEATURES']).to(DEVICE)
    
    if model_path is None:
        # Search for the latest model recursively
        all_models = list(C["PATHS"]["MODELS"].rglob("*.pth"))
        if all_models:
            model_path = max(all_models, key=lambda p: p.stat().st_mtime)
        else:
            model_path = C["PATHS"]["MODELS"] / 'best_model.pth' # fallback
            
    if model_path.exists():
        print(f"[*] Loading model weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    else:
        print("⚠️ Model weights not found! Using uninitialized model (will return garbage).")
        
    model.eval()
    
    print("[*] Running Inference...")
    with torch.no_grad():
        output = model(tensor_img)
        prob_mask = output.squeeze().cpu().numpy()
        bin_mask = (prob_mask > 0.5).astype(np.uint8) * 255
        
    # Plotting
    print("[*] Generating study plot...")
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    axs[0].imshow(original_rgb)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(preprocessed_rgb)
    axs[1].set_title('Post Preprocessing')
    axs[1].axis('off')
    
    axs[2].imshow(prob_mask, cmap='viridis')
    axs[2].set_title('Analog Prediction (Softmax/Sigmoid)')
    axs[2].axis('off')
    
    axs[3].imshow(bin_mask, cmap='gray')
    axs[3].set_title('Final Binarized Mask')
    axs[3].axis('off')
    
    plt.tight_layout()
    output_plot = C["PATHS"]["EVIDENCE"] / 'study_image_prediction.png'
    plt.savefig(output_plot, dpi=300)
    plt.close()
    
    print(f"✅ Inference complete. Study image saved to: {output_plot}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Segmentation Inference")
    parser.add_argument("--image", type=str, required=False, help="Path to input image")
    args = parser.parse_args()
    
    # Try picking a default image if none provided
    if not args.image:
        default_dir = PROJECT_ROOT / C["DATASET"]["BASE_PATH"] / C["DATASET"]["TEST_IMAGES"]
        if default_dir.exists():
            images = list(default_dir.glob("*.tif")) + list(default_dir.glob("*.png"))
            if images:
                args.image = str(images[0])
                
    if args.image:
        test_inference_on_image(Path(args.image))
    else:
        print("Provide an image path using --image argument")
