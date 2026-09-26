#!/usr/bin/env python3
"""
Generates the peripapillary Zone B figure used as Figure 7 of the thesis
(docs/thesis/): the fundus image with the 0.5 DD and 1.0 DD circles centered
on the optic disc found by the same hybrid detection the pipeline uses.

Follows the USP/Esalq MBA formatting manual: no title inside the figure
(the caption goes below it in the document), Arial (or the metric compatible
Liberation Sans when Arial is not installed), and decimal comma in the
Portuguese version.

Usage:
    python scripts/make_zone_b_figure.py --lang pt --out docs/thesis/figures/zone_b_pt.png
    python scripts/make_zone_b_figure.py --lang en --out docs/thesis/figures/zone_b_en.png
    python scripts/make_zone_b_figure.py --image data/DRIVE/test/images/02_test.tif --lang en --out /tmp/zone_b.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import OPTIC_DISC_CONFIG as OC, PIPELINE_CONFIG
from src.pipeline.integrated_pipeline import _find_latest_checkpoint
from src.pipeline.optic_disc import OpticDiscDetector, detect_optic_disc

LABELS = {
    "pt": {
        "inner": "Borda interna da Zona B (0,5 DD)",
        "outer": "Borda externa da Zona B (1,0 DD)",
        "center": "Centro do disco óptico detectado",
    },
    "en": {
        "inner": "Zone B inner edge (0.5 DD)",
        "outer": "Zone B outer edge (1.0 DD)",
        "center": "Detected optic disc center",
    },
}


def pick_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Liberation Sans"):
        if name in available:
            return name
    return "sans-serif"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", type=Path, default=Path("data/DRIVE/test/images/01_test.tif"))
    parser.add_argument("--lang", choices=["pt", "en"], default="pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Could not read {args.image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detector = None
    ckpt = _find_latest_checkpoint(OC["PATHS"]["MODELS"])
    if ckpt is not None:
        import torch
        detector = OpticDiscDetector(ckpt, device=torch.device(args.device), image_size=OC["DATASET"]["IMAGE_SIZE"])

    od = detect_optic_disc(image, model=detector, min_confidence=PIPELINE_CONFIG["OD_MIN_CONFIDENCE"])
    (cx, cy), radius = od["center"], od["radius"]
    disc_diameter = 2.0 * radius
    inner_r = PIPELINE_CONFIG["ZONE_B_INNER_DD"] * disc_diameter
    outer_r = PIPELINE_CONFIG["ZONE_B_OUTER_DD"] * disc_diameter
    print(f"{args.image.name}: method={od['method']} center=({cx:.1f}, {cy:.1f}) radius={radius:.1f}px")

    plt.rcParams["font.family"] = pick_font()
    plt.rcParams["font.size"] = 11
    labels = LABELS[args.lang]

    legend_height_in = 1.0
    width_in = 7.0
    image_height_in = width_in * image.shape[0] / image.shape[1]
    fig = plt.figure(figsize=(width_in, image_height_in + legend_height_in))
    bottom = legend_height_in / (image_height_in + legend_height_in)
    ax = fig.add_axes((0, bottom, 1, 1 - bottom))
    ax.imshow(image)
    for r, color in ((inner_r, "red"), (outer_r, "blue")):
        circle = plt.Circle((cx, cy), r, fill=False, color=color, linestyle="--", linewidth=2)
        ax.add_patch(circle)
        circle.set_clip_path(ax.patch)
    ax.plot(cx, cy, "o", color="lime", markersize=8, markeredgecolor="black")
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_axis_off()

    handles = [
        Line2D([0], [0], color="red", linestyle="--", linewidth=2, label=labels["inner"]),
        Line2D([0], [0], color="blue", linestyle="--", linewidth=2, label=labels["outer"]),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="lime", markeredgecolor="black",
               markersize=8, label=labels["center"]),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 0.0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
