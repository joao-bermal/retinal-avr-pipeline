#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trains the optic disc detection model (EnhancedUNet, 1 channel), used to
correctly build the peripapillary Zone B in the AVR calculation.

Small dataset (IOSTAR, 30 images with optic disc mask ground truth), so few
epochs. Equivalent to `python main.py --train_od`: same trainer settings
(including the 0.3 s pause per batch that keeps the GPU out of thermal
runaway, see docs/GPU_ROCM_RX6800XT.md) and the same center-error evidence
from scripts/evaluate_optic_disc.py.
"""

import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config.settings import OPTIC_DISC_CONFIG as C, DEVICE
from src.data.optic_disc_dataset import OpticDiscDataset
from src.data.segmentation_dataset import SegmentationAugmentation
from src.models.segmentation_model import EnhancedUNet
from src.training.segmentation_trainer import EnhancedSegmentationTrainer
from src.metrics.evaluation_metrics import compute_segmentation_metrics
from scripts.evaluate_optic_disc import run_evaluation


def evaluate_best_model_od(model, val_loader):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            p = model(x)
            if p.dim() == 4 and p.size(1) == 1:
                p = p.squeeze(1)
            all_metrics.append(compute_segmentation_metrics(p, y))
    dice, acc, sen, spe, iou = tuple(sum(m[i] for m in all_metrics) / len(all_metrics) for i in range(5))
    return {"dice_score": dice, "accuracy": acc, "sensitivity": sen, "specificity": spe, "iou": iou}


def main(epochs=None):
    print("=" * 60)
    print("OPTIC DISC DETECTION MODEL TRAINING")
    print("=" * 60)

    train_aug = SegmentationAugmentation(img_size=C["DATASET"]["IMAGE_SIZE"], phase="train").transform
    val_aug = SegmentationAugmentation(img_size=C["DATASET"]["IMAGE_SIZE"], phase="val").transform

    train_dataset = OpticDiscDataset(
        base_path=C["DATASET"]["BASE_PATH"], phase="train",
        img_size=C["DATASET"]["IMAGE_SIZE"], transform=train_aug,
    )
    val_dataset = OpticDiscDataset(
        base_path=C["DATASET"]["BASE_PATH"], phase="val",
        img_size=C["DATASET"]["IMAGE_SIZE"], transform=val_aug,
    )

    train_loader = DataLoader(train_dataset, batch_size=C["TRAINING"]["BATCH_SIZE"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=C["TRAINING"]["BATCH_SIZE"], shuffle=False, num_workers=0)

    model = EnhancedUNet(
        in_channels=C["MODEL"]["IN_CHANNELS"],
        out_channels=C["MODEL"]["OUT_CHANNELS"],
        features=C["MODEL"]["FEATURES"],
    ).to(DEVICE)

    trainer = EnhancedSegmentationTrainer(
        model, train_loader, val_loader, resume=False, config=C,
        keep_all_checkpoints=True, batch_pause_seconds=0.3,
    )

    print("\nStarting training...")
    best_model_path, history, final_metrics, run_id = trainer.train(epochs=epochs)

    if best_model_path and best_model_path.exists():
        print(f"\nLoading best checkpoint: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
        final_metrics = evaluate_best_model_od(model, val_loader)

        run_results_path = C["PATHS"]["RESULTS"] / run_id
        run_results_path.mkdir(parents=True, exist_ok=True)

        results = {
            "run_id": run_id,
            "best_dice_score": final_metrics["dice_score"],
            "target_achieved": final_metrics["dice_score"] >= C["TARGETS"]["DICE_SCORE"],
            "history": history,
            "final_metrics": final_metrics,
        }
        with open(run_results_path / "training_results.json", "w") as f:
            json.dump(results, f, indent=2)

        run_evaluation(best_model_path, run_results_path / "evidence", do_sweep=True)
        print(f"✅ Optic disc training complete. Dice: {final_metrics['dice_score']:.4f}")
        print(f"   Results saved to {run_results_path}")


if __name__ == "__main__":
    main()
