#!/usr/bin/env python3
"""
Evaluates optic disc detection on IOSTAR against the real disc masks
(data/IOSTAR/mask_OD) and writes the evidence used in docs/METRICS.md.

For every IOSTAR image it measures the distance, in pixels at native
resolution, between the ground-truth disc centroid and the center returned
by each method:

    trained_model   OpticDiscDetector with the given checkpoint
    cv_heuristic    detect_optic_disc_cv (no training)
    image_center    geometric center of the image (the original thesis assumption)
    hybrid          detect_optic_disc, the dispatcher the pipeline actually uses

The split is the same deterministic 80/20 split used by OpticDiscDataset
(first 24 sorted images train, last 6 held out), so "held_out" numbers are
images the model never saw during training.

Outputs (in --out, default results/optic_disc/<run_id>/evidence):
    per_image_center_error.csv   one row per image and method
    center_error_summary.csv     median / mean error per method and split
    center_error_by_method.png   bar chart of the held-out median error
    held_out_detections.png      the 6 held-out images with GT and predictions
    training_curves.png          Dice / loss / learning rate per epoch, read
                                 from the run's training_results.json
    epoch_sweep.csv              only with --sweep: held-out error of every
                                 checkpoint saved in the run directory

Usage:
    python scripts/evaluate_optic_disc.py
    python scripts/evaluate_optic_disc.py --checkpoint models/optic_disc/run_X/model_epoch039_dice_0.8545.pth
    python scripts/evaluate_optic_disc.py --sweep --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import OPTIC_DISC_CONFIG as OC, PIPELINE_CONFIG
from src.data.optic_disc_dataset import _extract_disc_mask
from src.pipeline.integrated_pipeline import _find_latest_checkpoint
from src.pipeline.optic_disc import OpticDiscDetector, detect_optic_disc, detect_optic_disc_cv

METHODS = ["trained_model", "cv_heuristic", "image_center", "hybrid"]


def list_iostar_pairs(base_path):
    image_dir = Path(base_path) / "image"
    mask_dir = Path(base_path) / "mask_OD"
    pairs = []
    for img_path in sorted(image_dir.glob("*.jpg")):
        mask_path = mask_dir / f"{img_path.stem}_ODMask.tif"
        if mask_path.exists():
            pairs.append((img_path, mask_path))
    if not pairs:
        raise SystemExit(f"No IOSTAR image/mask_OD pairs found under {base_path}")
    return pairs


def gt_center(mask_path):
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    disc = _extract_disc_mask(raw)
    ys, xs = np.nonzero(disc)
    return float(xs.mean()), float(ys.mean())


def predict_all(image_rgb, detector, min_confidence):
    h, w = image_rgb.shape[:2]
    preds = {
        "trained_model": detector.detect(image_rgb) if detector else None,
        "cv_heuristic": detect_optic_disc_cv(image_rgb),
        "image_center": {"center": (w / 2.0, h / 2.0), "method": "IMAGE_CENTER"},
        "hybrid": detect_optic_disc(image_rgb, model=detector, min_confidence=min_confidence),
    }
    return preds


def evaluate(pairs, split_idx, detector, min_confidence):
    rows = []
    for i, (img_path, mask_path) in enumerate(pairs):
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        gx, gy = gt_center(mask_path)
        preds = predict_all(image, detector, min_confidence)
        for method in METHODS:
            pred = preds[method]
            if pred is None:
                px = py = err = np.nan
                resolved = "NO_DETECTION"
            else:
                px, py = pred["center"]
                err = float(np.hypot(px - gx, py - gy))
                resolved = pred.get("method", method)
            rows.append({
                "image": img_path.name,
                "split": "train" if i < split_idx else "held_out",
                "method": method,
                "resolved_method": resolved,
                "gt_x": gx, "gt_y": gy, "pred_x": px, "pred_y": py,
                "center_error_px": err,
            })
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for split_name, sub in [("held_out", df[df.split == "held_out"]), ("all", df)]:
        for method in METHODS:
            errs = sub[sub.method == method]["center_error_px"]
            out.append({
                "split": split_name,
                "method": method,
                "n_images": int(errs.shape[0]),
                "n_no_detection": int(errs.isna().sum()),
                "median_error_px": float(errs.median()),
                "mean_error_px": float(errs.mean()),
            })
    return pd.DataFrame(out)


def plot_bar(summary, save_path):
    held = summary[summary.split == "held_out"].set_index("method").loc[METHODS]
    labels = ["Trained model", "CV heuristic", "Image center", "Hybrid"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, held["median_error_px"], color="0.35")
    for x, v in enumerate(held["median_error_px"]):
        ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Median center error on held-out images (px)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_held_out(df, pairs, split_idx, save_path):
    held_pairs = pairs[split_idx:]
    cols = 3
    rows = int(np.ceil(len(held_pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    style = {
        "trained_model": ("tab:blue", "o", "Trained model"),
        "cv_heuristic": ("tab:orange", "s", "CV heuristic"),
        "image_center": ("tab:red", "x", "Image center"),
    }
    for ax, (img_path, _) in zip(axes, held_pairs):
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        sub = df[df.image == img_path.name]
        gt = sub.iloc[0]
        ax.plot(gt.gt_x, gt.gt_y, marker="+", color="lime", markersize=18, mew=3, label="Ground truth")
        for method, (color, marker, label) in style.items():
            r = sub[sub.method == method].iloc[0]
            if not np.isnan(r.pred_x):
                ax.plot(r.pred_x, r.pred_y, marker=marker, color=color, markersize=9, mew=2,
                        linestyle="none", label=label)
        ax.set_xlabel(img_path.stem, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(held_pairs):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(results_json, save_path):
    history = json.loads(Path(results_json).read_text())["history"]
    epochs = range(1, len(history["train_dice"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    # The trainer stores 0.0 placeholders for train Dice (it is not computed
    # during training), so only plot it when real values are present.
    if any(v > 0 for v in history["train_dice"]):
        axes[0].plot(epochs, history["train_dice"], label="Train")
    axes[0].plot(epochs, history["val_dice"], label="Validation")
    axes[0].axhline(OC["TARGETS"]["DICE_SCORE"], color="g", linestyle="--", label="Target")
    axes[0].set_ylabel("Dice")
    axes[1].plot(epochs, history["train_loss"], label="Train")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    axes[1].set_ylabel("Combined loss")
    axes[2].plot(epochs, history["learning_rates"])
    axes[2].set_ylabel("Learning rate")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Epoch")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def sweep(run_dir, pairs, split_idx, device, min_confidence):
    rows = []
    for ckpt in sorted(Path(run_dir).glob("*.pth")):
        detector = OpticDiscDetector(ckpt, device=device, image_size=OC["DATASET"]["IMAGE_SIZE"])
        df = evaluate(pairs, split_idx, detector, min_confidence)
        held = df[(df.split == "held_out") & (df.method == "trained_model")]["center_error_px"]
        rows.append({"checkpoint": ckpt.name, "held_out_median_error_px": float(held.median()),
                     "held_out_no_detection": int(held.isna().sum())})
        print(f"  {ckpt.name}: held-out median error {held.median():.1f}px")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, help="Optic disc .pth (default: latest under models/optic_disc)")
    parser.add_argument("--out", type=Path, help="Output directory (default: results/optic_disc/<run_id>/evidence)")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    parser.add_argument("--sweep", action="store_true", help="Also score every checkpoint in the run directory")
    args = parser.parse_args()
    run_evaluation(args.checkpoint, args.out, args.device, args.sweep)


def run_evaluation(checkpoint=None, out_dir=None, device=None, do_sweep=False):
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = checkpoint or _find_latest_checkpoint(OC["PATHS"]["MODELS"])
    if checkpoint is None:
        raise SystemExit("No optic disc checkpoint found. Train one with: python main.py --train_od")
    checkpoint = Path(checkpoint)
    run_id = checkpoint.parent.name
    out_dir = Path(out_dir) if out_dir else (OC["PATHS"]["RESULTS"] / run_id / "evidence")
    out_dir.mkdir(parents=True, exist_ok=True)
    min_conf = PIPELINE_CONFIG["OD_MIN_CONFIDENCE"]

    pairs = list_iostar_pairs(OC["DATASET"]["BASE_PATH"])
    split_idx = max(1, int(len(pairs) * 0.8))
    print(f"Checkpoint: {checkpoint}")
    print(f"IOSTAR pairs: {len(pairs)} ({split_idx} train, {len(pairs) - split_idx} held out), device: {device}")

    detector = OpticDiscDetector(checkpoint, device=device, image_size=OC["DATASET"]["IMAGE_SIZE"])
    df = evaluate(pairs, split_idx, detector, min_conf)
    summary = summarize(df)

    df.to_csv(out_dir / "per_image_center_error.csv", index=False)
    summary.to_csv(out_dir / "center_error_summary.csv", index=False)
    plot_bar(summary, out_dir / "center_error_by_method.png")
    plot_held_out(df, pairs, split_idx, out_dir / "held_out_detections.png")

    results_json = OC["PATHS"]["RESULTS"] / run_id / "training_results.json"
    if results_json.exists():
        plot_training_curves(results_json, out_dir / "training_curves.png")

    if do_sweep:
        print("Epoch sweep:")
        sweep(checkpoint.parent, pairs, split_idx, device, min_conf).to_csv(out_dir / "epoch_sweep.csv", index=False)

    print("\nMedian center error (px):")
    print(summary.pivot(index="method", columns="split", values="median_error_px").loc[METHODS].round(1).to_string())
    print(f"\nEvidence written to {out_dir}")
    return summary


if __name__ == "__main__":
    main()
