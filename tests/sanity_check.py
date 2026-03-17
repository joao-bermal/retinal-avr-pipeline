import sys
import os
import torch

# Adicionar PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.segmentation_model import EnhancedUNet
from src.models.av_classification_model import EnhancedMultiDatasetAVNet
from src.config.settings import SEGMENTATION_CONFIG, AV_CLASSIFICATION_CONFIG

def test_segmentation_model():
    print("Testing Segmentation Model...")
    model = EnhancedUNet(
        in_channels=SEGMENTATION_CONFIG["MODEL"]["IN_CHANNELS"],
        out_channels=SEGMENTATION_CONFIG["MODEL"]["OUT_CHANNELS"]
    )
    # B, C, H, W (4, 3, 576, 608)
    dummy_input = torch.randn(2, 3, 576, 608)
    output = model(dummy_input)
    assert output.shape == (2, 1, 576, 608), f"Expected (2, 1, 576, 608) but got {output.shape}"
    print("✓ Segmentation Model forward pass successful!")

def test_av_classification_model():
    print("Testing AV Classification Model...")
    model = EnhancedMultiDatasetAVNet(AV_CLASSIFICATION_CONFIG)
    
    # B, C, H, W
    dummy_input = torch.randn(2, 3, 768, 768)
    
    # Optional mask
    dummy_mask = torch.randint(0, 2, (2, 768, 768)).float()
    
    output = model(dummy_input, vessel_mask=dummy_mask)
    assert output.shape == (2, 3, 768, 768), f"Expected (2, 3, 768, 768) but got {output.shape}"
    print("✓ AV Classification Model forward pass successful!")

if __name__ == "__main__":
    try:
        test_segmentation_model()
        test_av_classification_model()
        print("\nAll sanity checks passed successfully.")
    except Exception as e:
        print(f"\nSanity check failed: {e}")
        sys.exit(1)
