#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Treino do modelo de deteccao do disco optico (EnhancedUNet, 1 canal),
usado para corrigir a Zona B peripapilar no calculo do AVR.

Dataset pequeno (IOSTAR, ~30 imagens com mascara de disco optico) -- poucas
epocas, sem os graficos de evidencia pesados usados no treino de vasos.
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
    print("TREINO DO MODELO DE DETECCAO DO DISCO OPTICO")
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

    trainer = EnhancedSegmentationTrainer(model, train_loader, val_loader, resume=False, config=C)

    print("\nIniciando treino...")
    best_model_path, history, final_metrics, run_id = trainer.train(epochs=epochs)

    if best_model_path and best_model_path.exists():
        print(f"\nCarregando melhor checkpoint: {best_model_path}")
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

        print(f"✅ Treino do disco óptico concluído. Dice: {final_metrics['dice_score']:.4f}")
        print(f"   Resultados salvos em {run_results_path}")


if __name__ == "__main__":
    main()
