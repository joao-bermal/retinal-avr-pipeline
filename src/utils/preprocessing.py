import cv2, numpy as np

def apply_enhanced_preprocessing(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    processed = image.copy()
    green = processed[:, :, 1]
    
    # Preprocessing parameters from original notebook
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_SIZE = (8, 8)
    GAMMA_CORRECTION = 1.2
    
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    processed[:, :, 1] = clahe.apply(green)
    gamma = GAMMA_CORRECTION
    processed = np.power(processed / 255.0, gamma) * 255.0
    return processed.astype(np.uint8)
