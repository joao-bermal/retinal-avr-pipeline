"""
Unit tests for the scientific AVR calculation (src/pipeline/avr_calculator.py).

Uses synthetic masks (no dataset, no GPU) to validate:
1. The Zone B ring geometry (0.5-1.0 DD from the optic disc).
2. The Knudtson iterative combination (largest+smallest, not left-fold).
3. The full calculate_scientific_avr pipeline with known vessel widths.
"""

import importlib.util
import math
import os

import cv2
import numpy as np

# Loads avr_calculator.py directly from its file path, bypassing
# src.pipeline's __init__.py (which imports torch/torchvision for the rest
# of the pipeline) -- this module and its tests are pure
# numpy/cv2/scipy/skimage and shouldn't depend on a valid torch/torchvision
# installation.
_MODULE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "pipeline", "avr_calculator.py")
)
_spec = importlib.util.spec_from_file_location("avr_calculator", _MODULE_PATH)
avr_calculator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(avr_calculator)

ScientificAVRCalculator = avr_calculator.ScientificAVRCalculator
CRAE_COEFFICIENT = avr_calculator.CRAE_COEFFICIENT
CRVE_COEFFICIENT = avr_calculator.CRVE_COEFFICIENT


def test_zone_b_ring_geometry():
    """The ROI must include only the annulus between 0.5 DD and 1.0 DD from the disc center."""
    shape = (400, 400)
    center = (200, 200)
    radius = 40.0  # disc radius -> DD = 80px; Zone B = [40px, 80px]
    calc = ScientificAVRCalculator(shape, optic_disc_center=center, optic_disc_radius=radius)
    roi = calc._create_roi_mask()

    # Inside the disc (well below 0.5 DD): outside the ROI.
    assert roi[200, 210] == 0, "pixel inside the disc should not be in Zone B"
    # Halfway through Zone B (~60px from center): inside the ROI.
    assert roi[200, 260] == 1, "pixel in the middle of Zone B should be included"
    # Well beyond 1.0 DD: outside the ROI.
    assert roi[200, 320] == 0, "pixel beyond 1.0 DD should not be in Zone B"
    print("[OK] Zone B ring geometry (0.5-1.0 DD annulus)")


def test_knudtson_combination_uses_largest_smallest_pairing():
    """
    The combination must pair LARGEST with SMALLEST at every step, not a
    naive left-fold (the original notebook's bug).
    """
    widths = [10.0, 8.0, 6.0, 4.0]

    # Naive fold (old bug): always index 0 + index 1, re-inserted at the front.
    naive = widths.copy()
    while len(naive) > 1:
        w1, w2 = naive[0], naive[1]
        naive = [CRAE_COEFFICIENT * math.sqrt(w1 ** 2 + w2 ** 2)] + naive[2:]
    naive_result = naive[0]

    # Canonical algorithm: pair largest+smallest, re-sort, repeat.
    result = ScientificAVRCalculator._combine_knudtson(widths, CRAE_COEFFICIENT)

    # Manual calculation of the canonical result for [4,6,8,10]:
    # step 1: smallest=4, largest=10 -> c1 = 0.88*sqrt(10^2+4^2); remaining sorted [6,8,c1]
    c1 = CRAE_COEFFICIENT * math.sqrt(10.0 ** 2 + 4.0 ** 2)
    remaining = sorted([6.0, 8.0, c1])
    # step 2: smallest and largest of remaining
    smallest, largest = remaining[0], remaining[-1]
    mid = [w for w in remaining if w not in (smallest, largest)] or [remaining[1]]
    c2 = CRAE_COEFFICIENT * math.sqrt(largest ** 2 + smallest ** 2)
    expected = CRAE_COEFFICIENT * math.sqrt(max(c2, mid[0]) ** 2 + min(c2, mid[0]) ** 2)

    assert abs(result - expected) < 1e-6, f"expected {expected}, got {result}"
    assert abs(result - naive_result) > 1e-6, (
        "canonical result should not coincide with the naive fold in this case"
    )
    print(f"[OK] Knudtson largest-smallest pairing (canonical={result:.4f}, naive fold={naive_result:.4f})")


def test_knudtson_single_and_empty_width():
    assert ScientificAVRCalculator._combine_knudtson([], CRAE_COEFFICIENT) == 0.0
    assert ScientificAVRCalculator._combine_knudtson([5.0], CRAE_COEFFICIENT) == 5.0
    print("[OK] base cases (empty list / single caliber)")


def _draw_vessels(shape, center, count, angle_offset_deg, length, thickness):
    """Draws `count` straight lines radiating from the center, simulating radial vessels."""
    mask = np.zeros(shape, dtype=np.uint8)
    for i in range(count):
        angle = math.radians(angle_offset_deg + i * (360.0 / count))
        x0 = int(center[0] + 20 * math.cos(angle))
        y0 = int(center[1] + 20 * math.sin(angle))
        x1 = int(center[0] + length * math.cos(angle))
        y1 = int(center[1] + length * math.sin(angle))
        cv2.line(mask, (x0, y0), (x1, y1), color=255, thickness=thickness)
    return mask


def test_calculate_scientific_avr_with_known_widths():
    """
    Draws 6 thinner radial "arteries" and 6 thicker radial "veins" crossing
    Zone B, and checks that CRAE < CRVE and that the computed AVR is close
    to the nominal width ratio (Knudtson's exact ratio differs from a raw
    width/width ratio, so we only validate direction and order of magnitude).
    """
    shape = (600, 600)
    center = (300, 300)
    radius = 40.0  # Zone B = [40px, 80px] from center

    artery_thickness = 6   # px
    vein_thickness = 10    # px

    artery_mask = _draw_vessels(shape, center, count=6, angle_offset_deg=0, length=120, thickness=artery_thickness)
    vein_mask = _draw_vessels(shape, center, count=6, angle_offset_deg=30, length=120, thickness=vein_thickness)

    calc = ScientificAVRCalculator(shape, optic_disc_center=center, optic_disc_radius=radius)
    result = calc.calculate_scientific_avr(artery_mask, vein_mask)

    assert result["status"] == "SUCCESS", result
    assert result["crae"] > 0 and result["crve"] > 0
    assert result["crae"] < result["crve"], "thinner arteries should yield CRAE < CRVE"
    assert 0.0 < result["avr"] < 1.0
    assert result["measurements"]["artery_count"] >= 2
    assert result["measurements"]["vein_count"] >= 2
    print(f"[OK] full pipeline: AVR={result['avr']:.3f} CRAE={result['crae']:.2f} CRVE={result['crve']:.2f} "
          f"risk={result['risk_level']}")


def test_insufficient_vessels():
    """Without enough vessels in Zone B, must return INSUFFICIENT_VESSELS, not crash."""
    shape = (300, 300)
    empty_mask = np.zeros(shape, dtype=np.uint8)
    calc = ScientificAVRCalculator(shape, optic_disc_center=(150, 150), optic_disc_radius=20.0)
    result = calc.calculate_scientific_avr(empty_mask, empty_mask)
    assert result["status"] == "INSUFFICIENT_VESSELS"
    assert result["avr"] == 0.0
    print("[OK] insufficient-vessels fallback")


def test_fallback_center_without_real_optic_disc():
    """Without real od_center/od_radius, it still works (fallback), but must not crash."""
    shape = (400, 400)
    mask = _draw_vessels(shape, (200, 200), count=6, angle_offset_deg=0, length=150, thickness=8)
    calc = ScientificAVRCalculator(shape)  # no od_center/od_radius -> fallback with a warning
    assert calc.od_center == (200, 200)
    assert calc.od_radius == min(shape) * 0.15
    result = calc.calculate_scientific_avr(mask, mask)
    assert result["status"] in ("SUCCESS", "INSUFFICIENT_VESSELS")
    print("[OK] fallback without a real optic disc doesn't crash (but uses the documented heuristic)")


if __name__ == "__main__":
    try:
        test_zone_b_ring_geometry()
        test_knudtson_combination_uses_largest_smallest_pairing()
        test_knudtson_single_and_empty_width()
        test_calculate_scientific_avr_with_known_widths()
        test_insufficient_vessels()
        test_fallback_center_without_real_optic_disc()
        print("\nAll AVR calculator tests passed.")
    except Exception as e:
        print(f"\nTest failed: {e}")
        raise
