# src/pipeline/avr_calculator.py
"""
Scientific AVR (Arteriolar-to-Venular Ratio) calculation from binary
artery/vein masks, following the Knudtson protocol (peripapillary Zone B,
iterative CRAE/CRVE equivalence).

Ported from retinal-avr-cardiovascular-risk/notebooks/03_integrated_pipeline.ipynb
(ScientificAVRCalculator class), with two corrections relative to the
original:

1. The iterative CRAE/CRVE combination now follows the canonical Knudtson
   et al. (2003) algorithm -- at each step, combine the LARGEST with the
   SMALLEST remaining caliber, then re-sort -- instead of the original
   notebook's naive left-fold (always index 0 + index 1).
2. The optic disc center/radius used to build Zone B are now, in practice,
   required parameters: when omitted, the fallback to the image's geometric
   center now logs an explicit warning (it used to be silent). Real disc
   detection lives in src/pipeline/optic_disc.py.
"""

import logging

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)

# Knudtson protocol: Zone B = annulus between 0.5 DD and 1.0 DD of the optic disc.
ZONE_B_INNER_DD = 0.5
ZONE_B_OUTER_DD = 1.0

# Knudtson et al. (2003) equivalence coefficients.
CRAE_COEFFICIENT = 0.88
CRVE_COEFFICIENT = 0.95

MIN_VESSEL_WIDTH_PX = 2.0
MAX_VESSELS_PER_TYPE = 6
MIN_VESSELS_PER_TYPE = 2


class ScientificAVRCalculator:
    """
    Computes AVR from real vessel calibers measured in the peripapillary
    Zone B, using the Knudtson equivalence formulas.
    """

    def __init__(self, image_shape, optic_disc_center=None, optic_disc_radius=None):
        """
        Args:
            image_shape: shape (H, W[, C]) of the artery/vein mask.
            optic_disc_center: (x, y) in pixels. If None, falls back to the
                image's geometric center as a last resort (inaccurate --
                logs a warning). Always prefer passing the result of
                src.pipeline.optic_disc.detect_optic_disc().
            optic_disc_radius: disc radius in pixels. If None, estimated as
                15% of the smaller image dimension (rough heuristic, same
                caveat as above).
        """
        self.image_shape = image_shape
        h, w = image_shape[:2]

        if optic_disc_center is None:
            logger.warning(
                "ScientificAVRCalculator called without a real optic_disc_center -- "
                "falling back to the image's geometric center. This is NOT "
                "clinically valid; pass the result of detect_optic_disc()."
            )
            self.od_center = (w // 2, h // 2)
        else:
            self.od_center = optic_disc_center

        if optic_disc_radius is None:
            logger.warning(
                "ScientificAVRCalculator called without a real optic_disc_radius -- "
                "estimating it as 15%% of the smaller image dimension."
            )
            self.od_radius = min(h, w) * 0.15
        else:
            self.od_radius = optic_disc_radius

    def _create_roi_mask(self):
        """Zone B mask: annulus between 0.5 DD and 1.0 DD of the optic disc."""
        h, w = self.image_shape[:2]
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - self.od_center[0]) ** 2 + (y - self.od_center[1]) ** 2)

        disc_diameter_px = 2.0 * self.od_radius
        inner_radius = ZONE_B_INNER_DD * disc_diameter_px
        outer_radius = ZONE_B_OUTER_DD * disc_diameter_px

        roi_mask = (distances >= inner_radius) & (distances <= outer_radius)
        return roi_mask.astype(np.uint8)

    @staticmethod
    def _get_vessel_skeleton(vessel_mask, roi_mask):
        """Morphological skeleton of the vessels restricted to the ROI."""
        vessel_binary = (vessel_mask > 127).astype(np.uint8)
        vessel_in_roi = cv2.bitwise_and(vessel_binary, roi_mask)
        if vessel_in_roi.sum() == 0:
            return np.zeros_like(vessel_in_roi)
        return skeletonize(vessel_in_roi > 0).astype(np.uint8)

    @staticmethod
    def _measure_vessel_widths(vessel_mask, skeleton):
        """Width = 2x the distance from a skeleton point to the vessel edge."""
        vessel_binary = (vessel_mask > 127).astype(np.uint8)
        if vessel_binary.sum() == 0 or skeleton.sum() == 0:
            return []

        distance_map = distance_transform_edt(vessel_binary)
        ys, xs = np.where(skeleton > 0)
        widths = [2.0 * distance_map[y, x] for y, x in zip(ys, xs)]
        return [w for w in widths if w >= 1.0]

    @staticmethod
    def _combine_knudtson(widths, coefficient):
        """
        Canonical Knudtson iterative combination: at each step, combine the
        LARGEST with the SMALLEST remaining caliber and re-insert the
        combined value, until a single equivalent caliber remains.
        """
        if not widths:
            return 0.0
        if len(widths) == 1:
            return widths[0]

        remaining = sorted(widths)
        while len(remaining) > 1:
            smallest = remaining.pop(0)
            largest = remaining.pop(-1)
            combined = coefficient * np.sqrt(largest ** 2 + smallest ** 2)
            remaining.append(combined)
            remaining.sort()
        return remaining[0]

    def _calculate_crae(self, artery_widths):
        return self._combine_knudtson(artery_widths, CRAE_COEFFICIENT)

    def _calculate_crve(self, vein_widths):
        return self._combine_knudtson(vein_widths, CRVE_COEFFICIENT)

    @staticmethod
    def _interpret_avr(avr):
        """Clinical interpretation (Wong and Mitchell, 2003; Liew et al., 2023)."""
        if avr >= 0.67:
            return "NORMAL", "Low cardiovascular risk", "HIGH"
        elif avr >= 0.60:
            return "BORDERLINE", "Moderate risk - monitoring recommended", "MEDIUM"
        else:
            return "HIGH_RISK", "High cardiovascular risk - clinical evaluation needed", "HIGH"

    @staticmethod
    def _insufficient_vessels_result(artery_count, vein_count):
        return {
            "avr": 0.0,
            "crae": 0.0,
            "crve": 0.0,
            "status": "INSUFFICIENT_VESSELS",
            "risk_level": "INDETERMINATE",
            "risk_description": (
                f"Insufficient vessels in Zone B: {artery_count} arteries, "
                f"{vein_count} veins (minimum: {MIN_VESSELS_PER_TYPE} each)"
            ),
            "confidence": "LOW",
            "measurements": {
                "artery_count": artery_count,
                "vein_count": vein_count,
                "artery_widths": [],
                "vein_widths": [],
            },
            "method": "SCIENTIFIC_KNUDTSON",
        }

    def calculate_scientific_avr(self, artery_mask, vein_mask):
        """
        Computes the scientific AVR (Knudtson protocol) from binary artery
        and vein masks (0/255), restricted to the peripapillary Zone B.

        Returns:
            dict with avr, crae, crve, status, risk_level, risk_description,
            confidence, measurements and method.
        """
        try:
            roi_mask = self._create_roi_mask()

            artery_skeleton = self._get_vessel_skeleton(artery_mask, roi_mask)
            vein_skeleton = self._get_vessel_skeleton(vein_mask, roi_mask)

            artery_widths = self._measure_vessel_widths(artery_mask, artery_skeleton)
            vein_widths = self._measure_vessel_widths(vein_mask, vein_skeleton)

            artery_widths = [w for w in artery_widths if w >= MIN_VESSEL_WIDTH_PX]
            vein_widths = [w for w in vein_widths if w >= MIN_VESSEL_WIDTH_PX]

            top_arteries = sorted(artery_widths, reverse=True)[:MAX_VESSELS_PER_TYPE]
            top_veins = sorted(vein_widths, reverse=True)[:MAX_VESSELS_PER_TYPE]

            if len(top_arteries) < MIN_VESSELS_PER_TYPE or len(top_veins) < MIN_VESSELS_PER_TYPE:
                return self._insufficient_vessels_result(len(top_arteries), len(top_veins))

            crae = self._calculate_crae(top_arteries)
            crve = self._calculate_crve(top_veins)
            avr = crae / crve if crve > 0 else 0.0

            risk_level, risk_description, confidence = self._interpret_avr(avr)

            return {
                "avr": avr,
                "crae": crae,
                "crve": crve,
                "status": "SUCCESS",
                "risk_level": risk_level,
                "risk_description": risk_description,
                "confidence": confidence,
                "measurements": {
                    "artery_count": len(top_arteries),
                    "vein_count": len(top_veins),
                    "artery_widths": top_arteries,
                    "vein_widths": top_veins,
                    "roi_pixels": int(roi_mask.sum()),
                    "optic_disc_center": self.od_center,
                    "optic_disc_radius": self.od_radius,
                },
                "method": "SCIENTIFIC_KNUDTSON",
            }
        except Exception as e:  # noqa: BLE001 - we want a structured result, not a crash
            logger.exception("Scientific AVR calculation failed")
            return {
                "avr": 0.0,
                "crae": 0.0,
                "crve": 0.0,
                "status": "ERROR",
                "risk_level": "INDETERMINATE",
                "risk_description": f"Calculation error: {e}",
                "confidence": "LOW",
                "method": "SCIENTIFIC_KNUDTSON",
            }
