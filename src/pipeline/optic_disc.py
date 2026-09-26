# src/pipeline/optic_disc.py
"""
Optic disc (OD) detection, used to correctly localize the peripapillary
Zone B in the AVR calculation (see src/pipeline/avr_calculator.py).

Hybrid strategy:
1. `OpticDiscDetector`: trained model (EnhancedUNet reused, 1 output
   channel = disc probability), when a checkpoint is available under
   models/optic_disc/.
2. `detect_optic_disc_cv`: classical computer-vision heuristic (brightest
   region + largest connected component + minimum enclosing circle), always
   available, no training required.
3. `detect_optic_disc`: dispatches between the two, with a final fallback
   to the image's geometric center (the same old behavior, but now always
   logged explicitly instead of silently).
"""

import logging

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Same ImageNet normalization used in SegmentationAugmentation
# (src/data/segmentation_dataset.py) during optic disc detector training.
# It must match at inference time, or the model receives an input distribution
# completely different from what it saw during training.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# The optic disc typically occupies ~5% to ~25% of the smaller image
# dimension in standard fundus photographs (DRIVE/RITE/IOSTAR). Used to
# reject implausible detections.
MIN_DISC_RADIUS_FRACTION = 0.03
MAX_DISC_RADIUS_FRACTION = 0.25

# Last-resort fallback (the same heuristic documented in the thesis, p. 20).
FALLBACK_RADIUS_FRACTION = 0.15


def _bounds_for_shape(shape):
    h, w = shape[:2]
    min_dim = min(h, w)
    return min_dim * MIN_DISC_RADIUS_FRACTION, min_dim * MAX_DISC_RADIUS_FRACTION


def detect_optic_disc_cv(image_rgb):
    """
    Classical heuristic (no training): the optic disc is one of the
    brightest, most red/yellow-saturated regions of the fundus, and
    approximately circular.

    Args:
        image_rgb: RGB image (H, W, 3), uint8.

    Returns:
        dict {center: (x, y), radius: float, confidence: float, method: str}
        or None if no plausible region was found.
    """
    if image_rgb is None or image_rgb.size == 0:
        return None

    h, w = image_rgb.shape[:2]
    min_radius, max_radius = _bounds_for_shape(image_rgb.shape)

    # Red channel: the optic disc is typically the brightest region in the
    # red channel of a color fundus photograph (Zhu et al.; ARIA).
    red = image_rgb[:, :, 0].astype(np.float32)

    # Field-of-view (FOV) mask: ignore the black background outside the retina.
    fov_mask = (image_rgb.sum(axis=2) > 15).astype(np.uint8)
    if fov_mask.sum() < 0.05 * h * w:
        fov_mask = np.ones((h, w), dtype=np.uint8)

    red_blurred = cv2.GaussianBlur(red, (0, 0), sigmaX=min_dim_sigma(image_rgb.shape))
    red_blurred = cv2.bitwise_and(
        red_blurred.astype(np.uint8), red_blurred.astype(np.uint8), mask=fov_mask
    )

    # Top ~2% brightest pixels within the FOV.
    valid = red_blurred[fov_mask > 0]
    if valid.size == 0:
        return None
    threshold = np.percentile(valid, 98)
    bright_mask = ((red_blurred >= threshold) & (fov_mask > 0)).astype(np.uint8)

    # Morphological cleanup + largest connected component.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
    if num_labels <= 1:
        return None

    # Largest component (ignoring label 0 = background).
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    component_mask = (labels == largest_label).astype(np.uint8)

    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)

    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    area = cv2.contourArea(contour)
    circle_area = np.pi * radius ** 2
    circularity = area / circle_area if circle_area > 0 else 0.0

    if not (min_radius <= radius <= max_radius):
        logger.debug(
            "CV OD candidate rejected: radius %.1fpx outside plausible range [%.1f, %.1f]",
            radius, min_radius, max_radius,
        )
        return None

    confidence = float(np.clip(circularity, 0.0, 1.0))
    return {
        "center": (float(cx), float(cy)),
        "radius": float(radius),
        "confidence": confidence,
        "method": "CV_BRIGHTEST_REGION",
    }


def min_dim_sigma(shape):
    """Gaussian blur sigma, proportional to image size."""
    return max(3.0, min(shape[:2]) * 0.01)


class OpticDiscDetector:
    """
    Inference wrapper for an EnhancedUNet trained to segment the optic disc
    (1 output channel, same architecture as vessel segmentation, see
    src/models/segmentation_model.py).
    """

    def __init__(self, checkpoint_path, device=None, image_size=(512, 512), threshold=0.5):
        from src.models.segmentation_model import EnhancedUNet

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.threshold = threshold

        self.model = EnhancedUNet(in_channels=3, out_channels=1).to(self.device).eval()
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def detect(self, image_rgb):
        """
        Args:
            image_rgb: RGB image (H, W, 3), uint8, original size.

        Returns:
            dict {center, radius, confidence, method} in the original
            image's coordinate space, or None if the model found no
            plausible region.
        """
        h0, w0 = image_rgb.shape[:2]
        min_radius, max_radius = _bounds_for_shape(image_rgb.shape)

        resized = cv2.resize(image_rgb, (self.image_size[1], self.image_size[0]))
        normalized = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        prob_map = self.model(tensor).squeeze().cpu().numpy()
        mask = (prob_map > self.threshold).astype(np.uint8)

        if mask.sum() == 0:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)

        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        # Rescale back to the original image size.
        scale_x = w0 / self.image_size[1]
        scale_y = h0 / self.image_size[0]
        cx_orig, cy_orig = cx * scale_x, cy * scale_y
        radius_orig = radius * (scale_x + scale_y) / 2.0

        if not (min_radius <= radius_orig <= max_radius):
            logger.debug(
                "Model OD candidate rejected: radius %.1fpx outside plausible range",
                radius_orig,
            )
            return None

        confidence = float(np.clip(prob_map[mask > 0].mean(), 0.0, 1.0))
        return {
            "center": (float(cx_orig), float(cy_orig)),
            "radius": float(radius_orig),
            "confidence": confidence,
            "method": "TRAINED_MODEL",
        }


def detect_optic_disc(image_rgb, model=None, min_confidence=0.3):
    """
    Hybrid optic disc detection: tries the trained model (if provided and
    confident), falls back to the classical CV heuristic, and only as a
    last resort falls back to the image's geometric center (documented,
    non-clinical fallback).

    Args:
        image_rgb: RGB image (H, W, 3), uint8.
        model: optional OpticDiscDetector instance (None = skip straight to
            the classical CV heuristic).
        min_confidence: minimum confidence to accept a detection (from the
            model or from CV) before falling back to the next method.

    Returns:
        dict {center: (x, y), radius: float, confidence: float, method: str}
    """
    if model is not None:
        try:
            result = model.detect(image_rgb)
            if result is not None and result["confidence"] >= min_confidence:
                return result
            logger.info("Optic disc model has low confidence, trying classical CV.")
        except Exception:
            logger.exception("Failed to run the optic disc model, trying classical CV.")

    cv_result = detect_optic_disc_cv(image_rgb)
    if cv_result is not None and cv_result["confidence"] >= min_confidence:
        return cv_result

    h, w = image_rgb.shape[:2]
    logger.warning(
        "No optic disc detection method reached sufficient confidence, "
        "falling back to the image's geometric center (non-clinical "
        "fallback, documented as a limitation in the thesis)."
    )
    return {
        "center": (w / 2.0, h / 2.0),
        "radius": min(h, w) * FALLBACK_RADIUS_FRACTION,
        "confidence": 0.0,
        "method": "FALLBACK_IMAGE_CENTER",
    }
