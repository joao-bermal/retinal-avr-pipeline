#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to execute the Retinal A/V Classification model training.
Reproduces the behavior of notebook baselines while keeping files and metrics structured.
"""

import sys
import numpy as np
import torch
import json
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Setup paths to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config.settings import AV_CLASSIFICATION_CONFIG as C, DEVICE
from src.data.av_classification_dataset import CombinedAVDataset
from src.models.av_classification_model import EnhancedMultiDatasetAVNet
from src.training.av_classification_trainer import EnhancedMultiDatasetTrainer

from sklearn.metrics import f1_score, accuracy_score
from src.utils.metrics import (
    create_confusion_matrix_plot_av,
    create_precision_recall_curve_av,
    create_sample_predictions_plot_av,
    create_final_consolidated_analysis_av
)

def evaluate_best_model_av(model, val_loader):
    """Performs a final detailed evaluation with the best A/V model"""
    model.eval()
    all_logits = []
    all_targets = []
    
    print("\nExtracting final validation metrics (this may take a moment)...")
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch["image"].to(DEVICE), batch["mask"].to(DEVICE).long()
            logits = model(x)
            
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    val_logits = np.concatenate(all_logits, axis=0) # [N, 3, H, W]
    val_targets = np.concatenate(all_targets, axis=0) # [N, H, W]
    
    # Softmax across channel dimension (dim=1)
    # Using torch because it's fast
    val_probs = torch.softmax(torch.tensor(val_logits), dim=1).numpy()
    val_preds = np.argmax(val_probs, axis=1) # [N, H, W]
    
    # Flattening for Sklearn
    yt = val_targets.flatten()
    yp = val_preds.flatten()
    
    acc = accuracy_score(yt, yp)
    macro = f1_score(yt, yp, average="macro")
    f1c = f1_score(yt, yp, average=None, labels=[0,1,2])
    f1_art = float(f1c[1]) if len(f1c) > 1 else 0.0
    f1_vein = float(f1c[2]) if len(f1c) > 2 else 0.0
    
    return {
        'macro_f1': float(macro),
        'accuracy': float(acc),
        'f1_art': f1_art,
        'f1_vein': f1_vein,
        'probs': val_probs,
        'targets': val_targets
    }

def main():
    print("="*60)
    print("STARTING RETINAL A/V CLASSIFICATION PIPELINE (SCRIPTED)")
    print("="*60)
    
    # 1. Prepare Datasets and DataLoaders
    print("Loading Datasets...")
    train_dataset = CombinedAVDataset(C, phase="train")
    val_dataset = CombinedAVDataset(C, phase="val")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=C["TRAINING"]["BATCH_SIZE"],
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=C["TRAINING"]["BATCH_SIZE"],
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 2. Model
    model = EnhancedMultiDatasetAVNet(config=C).to(DEVICE)
    
    # 3. Training
    trainer = EnhancedMultiDatasetTrainer(
        model=model,
        cfg=C,
        device=DEVICE,
        resume=False
    )
    
    print("\nStarting Training...")
    best_model_path, history, final_metrics, run_id = trainer.fit(train_loader, val_loader)
    
    # 4. Final Evaluation & Plots
    if best_model_path and best_model_path.exists():
        print(f"\nLoading best A/V model for final metrics from {best_model_path}")
        # Trainer saves dictionary state under "state_dict"
        checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        
        final_metrics_complete = evaluate_best_model_av(model, val_loader)
        
        # Unique run paths
        run_results_path = C["PATHS"]["RESULTS"] / run_id
        evidence_path = run_results_path / "evidence"
        evidence_path.mkdir(parents=True, exist_ok=True)
        
        print("\nGenerating Scientific Plots and Evidence...")
        
        val_probs = final_metrics_complete.pop('probs')
        val_targets = final_metrics_complete.pop('targets')
        
        # 1. Confusion Matrix
        y_true_flat = val_targets.flatten()
        y_pred_flat = np.argmax(val_probs, axis=1).flatten()
        create_confusion_matrix_plot_av(
            y_true_flat, y_pred_flat, 
            save_path=evidence_path / 'confusion_matrix.png'
        )
        
        # 2. PR Curve
        # Convert multiclass into binary (Artery vs. Vein). Background (0) is ignored.
        mask_av = (y_true_flat > 0)
        yt_av = y_true_flat[mask_av] - 1  # Artery=0, Vein=1
        
        # We need probabilities for vein vs artery. The probabilities output are background(0), artery(1), vein(2).
        # We normalize prob_vein / (prob_artery + prob_vein) for active AV pixels.
        prob_art = val_probs[:, 1, :, :].flatten()[mask_av]
        prob_vein = val_probs[:, 2, :, :].flatten()[mask_av]
        yp_scores_av = prob_vein / (prob_art + prob_vein + 1e-8)
        
        create_precision_recall_curve_av(
            yt_av, yp_scores_av,
            save_path=evidence_path / 'pr_curve_av.png'
        )
        
        # 3. Sample Predictions Plot
        create_sample_predictions_plot_av(
            model, val_dataset, DEVICE,
            num_samples=3, save_path=evidence_path / 'sample_predictions.png'
        )
        
        # 4. Consolidated Analysis (equivalent to Fig03)
        # For av classification history, structure may be dict from list of dicts. Conversion needed.
        hist_df = pd.DataFrame(history)
        history_dict = hist_df.to_dict(orient='list')
        
        try:
            create_final_consolidated_analysis_av(
                final_metrics_complete, 
                save_path=evidence_path / 'consolidated_analysis.png'
            )
        except Exception as e:
            print(f"Warning: skipped consolidated plot due to error -> {e}")
        
        # Save results dictionary
        results = {
            'run_id': run_id,
            'best_macro_f1_score': final_metrics_complete['macro_f1'],
            'target_achieved': final_metrics_complete['macro_f1'] >= C['TARGETS']['MACRO_F1'],
            'history': history,
            'final_metrics': final_metrics_complete
        }
        
        results_path = run_results_path / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        metrics_df = pd.DataFrame([final_metrics_complete])
        metrics_df.to_csv(evidence_path / 'final_metrics.csv', index=False)
        
        print("✅ A/V Classification training complete. All evidence has been generated!")

if __name__ == "__main__":
    main()
