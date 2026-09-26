#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to execute the Vessel Segmentation model training.
Reproduces the behavior of 01_retinal_vessel_segmentation.ipynb while keeping
the files and metrics structured.
"""

import sys
import numpy as np
import torch
import json
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

# Setup paths to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config.settings import SEGMENTATION_CONFIG as C, DEVICE
from src.data.segmentation_dataset import EnhancedDRIVEDataset, SegmentationAugmentation
from src.models.segmentation_model import EnhancedUNet
from src.training.segmentation_trainer import EnhancedSegmentationTrainer
from src.utils.metrics import (
    create_metrics_visualization_segmentation,
    create_comprehensive_training_analysis_segmentation,
    create_sample_predictions_plot_segmentation,
    create_architecture_analysis_complete_segmentation
)
from src.metrics.evaluation_metrics import compute_segmentation_metrics

def evaluate_best_model(model, val_loader):
    """Performs a final detailed evaluation with the best model"""
    model.eval()
    all_metrics = []
    total_loss = 0.0
    from src.training.losses import CombinedLossOptimized
    criterion = CombinedLossOptimized().to(DEVICE)
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            p = model(x)
            
            loss_out = criterion(p, y)
            loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
            total_loss += loss.item()
            
            if p.dim() == 4 and p.size(1) == 1:
                p = p.squeeze(1)
            
            all_probs.append(p.cpu().numpy().flatten())
            all_targets.append(y.cpu().numpy().flatten())
            all_metrics.append(compute_segmentation_metrics(p, y))
            
    avg_loss = total_loss / len(val_loader)
    avg_metrics = tuple(sum(m[i] for m in all_metrics)/len(all_metrics) for i in range(5))
    dice, acc, sen, spe, iou = avg_metrics
    val_probs = np.concatenate(all_probs)
    val_targets = np.concatenate(all_targets)
    
    return {
        'total_loss': avg_loss,
        'dice_score': dice,
        'accuracy': acc,
        'sensitivity': sen,
        'specificity': spe,
        'iou': iou,
        'probs': val_probs,
        'targets': val_targets
    }

def main():
    print("="*60)
    print("STARTING RETINAL VESSEL SEGMENTATION PIPELINE (SCRIPTED)")
    print("="*60)
    
    # 1. Preparar Datasets e DataLoaders
    print("Loading Datasets...")
    train_aug = SegmentationAugmentation(phase="train").transform
    val_aug = SegmentationAugmentation(phase="val").transform
    
    train_dataset = EnhancedDRIVEDataset(
        base_path=C["DATASET"]["BASE_PATH"], 
        phase="train", 
        transform=train_aug
    )
    val_dataset = EnhancedDRIVEDataset(
        base_path=C["DATASET"]["BASE_PATH"], 
        phase="val", 
        transform=val_aug
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=C["TRAINING"]["BATCH_SIZE"],
        shuffle=True, 
        num_workers=0, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=C["TRAINING"]["BATCH_SIZE"],
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )
    
    # 2. Modelo
    model = EnhancedUNet(
        in_channels=C["MODEL"]["IN_CHANNELS"], 
        out_channels=C["MODEL"]["OUT_CHANNELS"],
        features=C["MODEL"]["FEATURES"]
    ).to(DEVICE)
    
    # 3. Treinamento
    trainer = EnhancedSegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        resume=False
    )
    
    print("\nStarting Training...")
    best_model_path, history, final_metrics, run_id = trainer.train()
    
    # 4. Avaliação e Gráficos Finais
    if best_model_path and best_model_path.exists():
        print(f"\nLoading best model for final metrics from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
        
        final_metrics = evaluate_best_model(model, val_loader)
        
        # Unique run paths
        run_results_path = C["PATHS"]["RESULTS"] / run_id
        evidence_path = run_results_path / "evidence"
        evidence_path.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating Scientific Plots and Evidence...")
        # Visualizações
        create_metrics_visualization_segmentation(history, save_path=evidence_path / 'training_curves_final.png')
        create_comprehensive_training_analysis_segmentation(history, final_metrics, save_path=evidence_path / 'fig03_training_curves_complete.png')
        
        # Testando sample predictions
        create_sample_predictions_plot_segmentation(model, val_dataset, DEVICE, num_samples=3, save_path=evidence_path / 'sample_predictions.png')
        
        # Análise arquitetônica
        create_architecture_analysis_complete_segmentation(
            model, 
            val_probs=final_metrics.pop('probs', None),
            val_targets=final_metrics.pop('targets', None),
            save_path=evidence_path / 'fig02_architecture_analysis.png'
        )
        
        # Salvar resultados e métricas em JSON e CSV
        results = {
            'run_id': run_id,
            'best_dice_score': final_metrics['dice_score'],
            'target_achieved': final_metrics['dice_score'] >= C['TARGETS']['DICE_SCORE'],
            'history': history,
            'final_metrics': final_metrics
        }
        
        results_path = run_results_path / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        metrics_df = pd.DataFrame([final_metrics])
        metrics_df.to_csv(evidence_path / 'final_metrics.csv', index=False)
        
        print("✅ Segmentation training complete. All evidence has been generated!")

if __name__ == "__main__":
    main()
