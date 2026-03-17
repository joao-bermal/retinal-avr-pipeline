import cv2
import numpy as np

def harmonize_av_data(image, av_mask, target_size=768):
    """
    Harmonize image and A/V mask from different datasets
    
    Args:
        image: Input fundus image (any format)
        av_mask: A/V ground truth mask (any encoding)
        target_size: Target size for harmonization
    
    Returns:
        image: Normalized RGB image
        av_mask: Standardized A/V mask (0=bg, 1=artery, 2=vein)
    """
    # Ensure image is RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Already RGB, just resize
        image = cv2.resize(image, (target_size, target_size))
    elif len(image.shape) == 2:
        # Grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (target_size, target_size))
    else:
        # Handle other formats
        image = cv2.resize(image, (target_size, target_size))
    
    # Ensure mask is 2D
    if len(av_mask.shape) == 3:
        av_mask = av_mask[:, :, 0]  # Take first channel
    
    # Resize mask
    av_mask = cv2.resize(av_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    
    return image, av_mask

def standardize_av_mask_rite(av_mask):
    """Standardize RITE A/V mask - baseado no processamento IOSTAR que funciona"""
    h, w = av_mask.shape[:2]
    classes = np.zeros((h, w), dtype=np.uint8)
    
    # RITE uses RGB colors in PNG format
    if len(av_mask.shape) == 3:
        r = av_mask[:, :, 0].astype(np.float32)
        g = av_mask[:, :, 1].astype(np.float32)
        b = av_mask[:, :, 2].astype(np.float32)
        
        tolerance = 30  # Slightly higher tolerance for RITE
        
        # Artérias (vermelho)
        artery_mask = (r >= 255 - tolerance) & (g <= tolerance) & (b <= tolerance)
        # Veias (azul)  
        vein_mask = (b >= 255 - tolerance) & (r <= tolerance) & (g <= tolerance)
        
        classes[artery_mask] = 1  # Artéria
        classes[vein_mask] = 2    # Veia
        
        # Alternative approach if no vessels detected
        artery_pixels = np.sum(artery_mask)
        vein_pixels = np.sum(vein_mask)
        
        if artery_pixels + vein_pixels < 50:  # Very few pixels detected
            # Try different approach: any colored pixels
            non_black = (r > 20) | (g > 20) | (b > 20)
            
            # Check for different color patterns
            # Method 1: Use intensity levels
            gray = (r + g + b) / 3
            unique_vals = np.unique(gray[non_black])
            
            if len(unique_vals) > 1:
                # Split by intensity
                median_val = np.median(unique_vals)
                low_intensity = (gray < median_val) & non_black
                high_intensity = (gray >= median_val) & non_black
                
                classes[low_intensity] = 1   # Artéria (darker)
                classes[high_intensity] = 2  # Veia (brighter)
            else:
                # Fallback: use red vs blue dominance
                red_dominant = (r > b) & (r > g) & non_black
                blue_dominant = (b > r) & (b > g) & non_black
                
                classes[red_dominant] = 1  # Artéria  
                classes[blue_dominant] = 2   # Veia
    
    return classes

def standardize_av_mask_iostar(av_mask):
    """
    Standardize IOSTAR A/V mask encoding
    IOSTAR: RGB color-coded (custom encoding)
    """
    h, w = av_mask.shape[:2]
    standardized = np.zeros((h, w), dtype=np.uint8)
    
    if len(av_mask.shape) == 3:
        # Assumindo encoding específico do IOSTAR
        # Você pode ajustar baseado no formato real
        gray = cv2.cvtColor(av_mask, cv2.COLOR_RGB2GRAY)
        
        # Thresholding baseado em intensidade
        # Ajustar baseado no formato real do IOSTAR
        artery_mask = (gray > 64) & (gray < 128)
        vein_mask = gray > 128
        
        standardized[artery_mask] = 1  # Artéria
        standardized[vein_mask] = 2    # Veia
    
    return standardized

def create_lesav_mask(artery_path, vein_path, target_size=768):
    """
    Create combined A/V mask from LES-AV separate artery and vein masks
    """
    artery_mask = cv2.imread(str(artery_path), cv2.IMREAD_GRAYSCALE)
    vein_mask = cv2.imread(str(vein_path), cv2.IMREAD_GRAYSCALE)
    
    # Resize both masks
    artery_mask = cv2.resize(artery_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    vein_mask = cv2.resize(vein_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    
    # Create combined mask
    combined = np.zeros_like(artery_mask, dtype=np.uint8)
    combined[artery_mask > 128] = 1  # Artéria
    combined[vein_mask > 128] = 2    # Veia
    
    return combined

def create_vessel_mask_from_av(av_mask):
    """
    Create binary vessel mask from A/V mask
    
    Args:
        av_mask: A/V mask (0=bg, 1=artery, 2=vein)
    
    Returns:
        vessel_mask: Binary vessel mask (0=bg, 1=vessel)
    """
    return ((av_mask == 1) | (av_mask == 2)).astype(np.uint8)