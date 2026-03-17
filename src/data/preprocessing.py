import cv2
import numpy as np
from src.config.settings import SEGMENTATION_CONFIG as C

def apply_enhanced_preprocessing(image: np.ndarray, visualize: bool = False) -> np.ndarray:
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    processed = image.copy()
    green = processed[:, :, 1]
    
    # Optional dynamic limits
    clip_limit = C.get("CLAHE_CLIP_LIMIT", 2.0) if isinstance(C, dict) else C.CLAHE_CLIP_LIMIT
    tile_size = C.get("CLAHE_TILE_SIZE", (8, 8)) if isinstance(C, dict) else C.CLAHE_TILE_SIZE
    gamma_val = C.get("GAMMA_CORRECTION", 1.2) if isinstance(C, dict) else C.GAMMA_CORRECTION
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    processed[:, :, 1] = clahe.apply(green)
    processed = np.power(processed / 255.0, gamma_val) * 255.0
    return processed.astype(np.uint8)
