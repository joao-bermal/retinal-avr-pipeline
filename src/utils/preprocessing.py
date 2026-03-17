import cv2, numpy as np
from src.configs.segmentation_config import EnhancedSegmentationConfig as C

def apply_enhanced_preprocessing(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    processed = image.copy()
    green = processed[:, :, 1]
    clahe = cv2.createCLAHE(clipLimit=C.CLAHE_CLIP_LIMIT, tileGridSize=C.CLAHE_TILE_SIZE)
    processed[:, :, 1] = clahe.apply(green)
    gamma = C.GAMMA_CORRECTION
    processed = np.power(processed / 255.0, gamma) * 255.0
    return processed.astype(np.uint8)
