#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script for Inference and Evaluation (Study Images) for A/V Classification.
Allows taking a random image, applying preprocessing, and extracting Artery/Vein predictions.
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

from src.config.settings import AV_CLASSIFICATION_CONFIG as C, DEVICE
from src.data.preprocessing import apply_enhanced_preprocessing
from src.models.av_classification_model import EnhancedMultiDatasetAVNet

def test_inference_on_av(image_path: Path, model_path: Path = None):
    """
    Runs inference of a trained A/V model on an image, and outputs Artery and Vein probability maps.
    """
    print(f"[*] Loading image from: {image_path}")
    if not image_path.exists():
        print("❌ Image not found!")
        return

    # Original Image
    original_bgr = cv2.imread(str(image_path))
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # 1. Preprocessing
    print("[*] Applying preprocessing...")
    preprocessed_rgb = apply_enhanced_preprocessing(original_rgb)
    
    # Resize to model size
    img_size = C['DATASET']['IMAGE_SIZE']
    resized_rgb = cv2.resize(preprocessed_rgb, (img_size, img_size))
    
    # Transform to tensor
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    normalized = (resized_rgb.astype(np.float32) / 255.0 - mean) / std
    tensor_img = torch.tensor(normalized).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    # Load Model (disable pretraining for inference load)
    C["MODEL"]["PRETRAINED"] = False
    model = EnhancedMultiDatasetAVNet(config=C).to(DEVICE)
    
    if model_path is None:
        # Search for the latest model recursively
        all_models = list(C["PATHS"]["MODELS"].rglob("*.pth"))
        if all_models:
            model_path = max(all_models, key=lambda p: p.stat().st_mtime)
        else:
            model_path = C["PATHS"]["MODELS"] / 'best_model.pth' # fallback
            
    if model_path.exists():
        print(f"[*] Loading A/V model weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("⚠️ Model weights not found! Using uninitialized model (will return garbage).")
        
    model.eval()
    
    print("[*] Running A/V Inference...")
    with torch.no_grad():
        logits = model(tensor_img)
        # Softmax to get probabilities (Background, Artery, Vein)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        prob_bg = probs[0]
        prob_art = probs[1]
        prob_vein = probs[2]
        
    # Plotting
    print("[*] Generating A/V study plot...")
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    axs[0].imshow(original_rgb)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(prob_art, cmap='Reds')
    axs[1].set_title('Artery Probabilities')
    axs[1].axis('off')
    
    axs[2].imshow(prob_vein, cmap='Blues')
    axs[2].set_title('Vein Probabilities')
    axs[2].axis('off')
    
    # Combined Image (Red array, Blue vein, Background logic)
    combined = np.zeros((*prob_art.shape, 3), dtype=np.float32)
    combined[..., 0] = prob_art # Red channel
    combined[..., 2] = prob_vein # Blue channel
    
    axs[3].imshow(combined)
    axs[3].set_title('A/V Overlay')
    axs[3].axis('off')
    
    plt.tight_layout()
    output_dir = C["PATHS"]["RESULTS"] / 'evidence'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_plot = output_dir / 'study_av_prediction.png'
    plt.savefig(output_plot, dpi=300)
    plt.close()
    
    print(f"✅ Inference complete. Study image saved to: {output_plot}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test A/V Classification Inference")
    parser.add_argument("--image", type=str, required=False, help="Path to input image")
    args = parser.parse_args()
    
    if not args.image:
        # Default to DRIVE or IOSTAR if possible
        default_dir = PROJECT_ROOT / "data" / "DRIVE" / "test" / "images"
        if default_dir.exists():
            images = list(default_dir.glob("*.tif")) + list(default_dir.glob("*.png"))
            if images:
                args.image = str(images[0])
                
    if args.image:
        test_inference_on_av(Path(args.image))
    else:
        print("Provide an image path using --image argument")
