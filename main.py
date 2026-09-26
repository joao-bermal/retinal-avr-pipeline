import torch
import numpy as np
import argparse
import os
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from src.training.segmentation_trainer import EnhancedSegmentationTrainer
from src.training.av_classification_trainer import EnhancedMultiDatasetTrainer
from src.pipeline.integrated_pipeline import ScientificAVRPipeline
from src.config.settings import SEGMENTATION_CONFIG, AV_CLASSIFICATION_CONFIG, OPTIC_DISC_CONFIG, PIPELINE_CONFIG, DEVICE
from src.data.segmentation_dataset import EnhancedDRIVEDataset, SegmentationAugmentation
from src.data.av_classification_dataset import EnhancedIOSTARDataset, CombinedAVDataset
from src.data.optic_disc_dataset import OpticDiscDataset
from src.models.segmentation_model import EnhancedUNet
from src.models.av_classification_model import EnhancedMultiDatasetAVNet

from scripts.train_optic_disc import evaluate_best_model_od

from scripts.train_segmentation import evaluate_best_model
from scripts.train_classification import evaluate_best_model_av

from src.utils.metrics import (
    create_metrics_visualization_segmentation,
    create_comprehensive_training_analysis_segmentation,
    create_sample_predictions_plot_segmentation,
    create_architecture_analysis_complete_segmentation,
    
    create_confusion_matrix_plot_av,
    create_precision_recall_curve_av,
    create_sample_predictions_plot_av,
    create_final_consolidated_analysis_av
)
import json
import pandas as pd

def update_config_from_yaml(config, yaml_path):
    if not yaml_path or not os.path.exists(yaml_path): return config
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
        for k, v in yaml_config.items():
            if k in config and isinstance(config[k], dict) and isinstance(v, dict):
                config[k].update(v)
            else:
                config[k] = v
    return config

def main():
    parser = argparse.ArgumentParser(description="Retinal Image Analysis Pipeline")
    parser.add_argument("--train_seg", action="store_true", help="Train the vessel segmentation model.")
    parser.add_argument("--train_av", action="store_true", help="Train the A/V classification model.")
    parser.add_argument("--train_od", action="store_true", help="Train the optic disc detection model.")
    parser.add_argument("--run_pipeline", type=str, help="Run the integrated pipeline on a specific image.")
    parser.add_argument("--resume", action="store_true", help="Resume training from an existing checkpoint.")
    parser.add_argument("--config", type=str, help="Path to a .yaml file to override default configurations in settings.py.")
    parser.add_argument("--lr", type=float, help="Override the Learning Rate.")
    parser.add_argument("--epochs", type=int, help="Override the number of Epochs.")
    parser.add_argument("--batch_size", type=int, help="Override the Batch Size.")
    
    args = parser.parse_args()

    # Limit VRAM usage to avoid system crash
    if torch.cuda.is_available():
        torch.cuda.memory.set_per_process_memory_fraction(0.85)
        # cudnn/MIOpen benchmark mode re-tunes the conv algorithm on the first
        # occurrence of each new shape. On CUDA this is a quick, worthwhile
        # trade. On this ROCm/MIOpen setup (pip wheels, no prebuilt perf-db
        # for gfx1030) it degenerates into an exhaustive per-shape search
        # that can take many minutes per layer -- confirmed empirically.
        torch.backends.cudnn.benchmark = torch.version.hip is None
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB) — VRAM limited to 85%")
    else:
        print("⚠️  No GPU detected, using CPU")

    # Apply configuration overrides
    if args.config:
        update_config_from_yaml(SEGMENTATION_CONFIG, args.config)
        update_config_from_yaml(AV_CLASSIFICATION_CONFIG, args.config)
    
    # Direct overrides
    if args.lr:
        SEGMENTATION_CONFIG["TRAINING"]["LEARNING_RATE"] = args.lr
        AV_CLASSIFICATION_CONFIG["TRAINING"]["LEARNING_RATE"] = args.lr
        OPTIC_DISC_CONFIG["TRAINING"]["LEARNING_RATE"] = args.lr
    if args.epochs:
        SEGMENTATION_CONFIG["TRAINING"]["EPOCHS"] = args.epochs
        AV_CLASSIFICATION_CONFIG["TRAINING"]["EPOCHS"] = args.epochs
        OPTIC_DISC_CONFIG["TRAINING"]["EPOCHS"] = args.epochs
    if args.batch_size:
        SEGMENTATION_CONFIG["TRAINING"]["BATCH_SIZE"] = args.batch_size
        AV_CLASSIFICATION_CONFIG["TRAINING"]["BATCH_SIZE"] = args.batch_size
        OPTIC_DISC_CONFIG["TRAINING"]["BATCH_SIZE"] = args.batch_size

    # Ensure output directories exist
    for path_config in [SEGMENTATION_CONFIG["PATHS"], AV_CLASSIFICATION_CONFIG["PATHS"], OPTIC_DISC_CONFIG["PATHS"], {"OUTPUT_DIR": PIPELINE_CONFIG["OUTPUT_DIR"]}]:
        for key, path_obj in path_config.items():
            if isinstance(path_obj, Path):
                path_obj.mkdir(parents=True, exist_ok=True)

    if args.train_seg:
        print("\n=== STARTING VESSEL SEGMENTATION TRAINING ===")
        # Build transforms and loaders
        train_aug = SegmentationAugmentation(img_size=SEGMENTATION_CONFIG["DATASET"]["IMAGE_SIZE"], phase="train")
        val_aug = SegmentationAugmentation(img_size=SEGMENTATION_CONFIG["DATASET"]["IMAGE_SIZE"], phase="val")
        
        train_ds = EnhancedDRIVEDataset(SEGMENTATION_CONFIG["DATASET"]["BASE_PATH"], phase="train", transform=train_aug.transform)
        val_ds = EnhancedDRIVEDataset(SEGMENTATION_CONFIG["DATASET"]["BASE_PATH"], phase="test", transform=val_aug.transform)
        
        train_loader = DataLoader(train_ds, batch_size=SEGMENTATION_CONFIG["TRAINING"]["BATCH_SIZE"], shuffle=True, num_workers=2, pin_memory=False)
        val_loader = DataLoader(val_ds, batch_size=SEGMENTATION_CONFIG["TRAINING"]["BATCH_SIZE"], shuffle=False, num_workers=2, pin_memory=False)
        
        model = EnhancedUNet(in_channels=SEGMENTATION_CONFIG["MODEL"]["IN_CHANNELS"], out_channels=SEGMENTATION_CONFIG["MODEL"]["OUT_CHANNELS"])
        trainer = EnhancedSegmentationTrainer(model, train_loader, val_loader, resume=args.resume)
        
        best_model_path, history, final_metrics, run_id = trainer.train(epochs=SEGMENTATION_CONFIG["TRAINING"]["EPOCHS"])
        
        # Generate scientific evidence plots locally in run container
        if best_model_path and best_model_path.exists():
            print(f"\nLoading best model for final metrics from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
            
            final_metrics = evaluate_best_model(model, val_loader)
            
            run_results_path = SEGMENTATION_CONFIG["PATHS"]["RESULTS"] / run_id
            evidence_path = run_results_path / "evidence"
            evidence_path.mkdir(parents=True, exist_ok=True)
            
            print("\nGenerating Scientific Plots and Evidence...")
            create_metrics_visualization_segmentation(history, save_path=evidence_path / 'training_curves_final.png')
            create_comprehensive_training_analysis_segmentation(history, final_metrics, save_path=evidence_path / 'fig03_training_curves_complete.png')
            create_sample_predictions_plot_segmentation(model, val_ds, DEVICE, num_samples=3, save_path=evidence_path / 'sample_predictions.png')
            
            create_architecture_analysis_complete_segmentation(
                model, 
                val_probs=final_metrics.pop('probs', None),
                val_targets=final_metrics.pop('targets', None),
                save_path=evidence_path / 'fig02_architecture_analysis.png'
            )
            
            # Save results dictionary
            results = {
                'run_id': run_id,
                'best_dice_score': final_metrics['dice_score'],
                'target_achieved': final_metrics['dice_score'] >= SEGMENTATION_CONFIG['TARGETS']['DICE_SCORE'],
                'history': history,
                'final_metrics': final_metrics
            }
            with open(run_results_path / 'training_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            pd.DataFrame([final_metrics]).to_csv(evidence_path / 'final_metrics.csv', index=False)
            print(f"✅ Evidence saved to {run_results_path}")

        print("=== SEGMENTATION TRAINING COMPLETED ===\n")
    
    if args.train_av:
        print("\n=== STARTING A/V CLASSIFICATION TRAINING ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_ds = CombinedAVDataset(AV_CLASSIFICATION_CONFIG, phase='train')
        val_ds = CombinedAVDataset(AV_CLASSIFICATION_CONFIG, phase='val')
        
        train_loader = DataLoader(train_ds, batch_size=AV_CLASSIFICATION_CONFIG["TRAINING"]["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=AV_CLASSIFICATION_CONFIG["TRAINING"]["BATCH_SIZE"], shuffle=False, num_workers=4, pin_memory=True)
        
        model = EnhancedMultiDatasetAVNet(config=AV_CLASSIFICATION_CONFIG).to(device)
        
        trainer = EnhancedMultiDatasetTrainer(model, AV_CLASSIFICATION_CONFIG, device, resume=args.resume)
        
        best_model_path, history, final_metrics, run_id = trainer.fit(train_loader, val_loader)
        
        if best_model_path and best_model_path.exists():
            print(f"\nLoading best A/V model for final metrics from {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            
            final_metrics_complete = evaluate_best_model_av(model, val_loader)
            
            run_results_path = AV_CLASSIFICATION_CONFIG["PATHS"]["RESULTS"] / run_id
            evidence_path = run_results_path / "evidence"
            evidence_path.mkdir(parents=True, exist_ok=True)
            
            print("\nGenerating Scientific Plots and Evidence...")
            
            val_probs = final_metrics_complete.pop('probs')
            val_targets = final_metrics_complete.pop('targets')
            
            # Confusion Matrix
            y_true_flat = val_targets.flatten()
            y_pred_flat = np.argmax(val_probs, axis=1).flatten()
            create_confusion_matrix_plot_av(y_true_flat, y_pred_flat, save_path=evidence_path / 'confusion_matrix.png')
            
            # PR Curve
            mask_av = (y_true_flat > 0)
            yt_av = y_true_flat[mask_av] - 1
            prob_art = val_probs[:, 1, :, :].flatten()[mask_av]
            prob_vein = val_probs[:, 2, :, :].flatten()[mask_av]
            yp_scores_av = prob_vein / (prob_art + prob_vein + 1e-8)
            create_precision_recall_curve_av(yt_av, yp_scores_av, save_path=evidence_path / 'pr_curve_av.png')
            
            create_sample_predictions_plot_av(model, val_ds, device, num_samples=3, save_path=evidence_path / 'sample_predictions.png')
            
            try:
                create_final_consolidated_analysis_av(final_metrics_complete, save_path=evidence_path / 'consolidated_analysis.png')
            except Exception as e:
                pass
                
            results = {
                'run_id': run_id,
                'best_macro_f1_score': final_metrics_complete['macro_f1'],
                'target_achieved': final_metrics_complete['macro_f1'] >= AV_CLASSIFICATION_CONFIG['TARGETS']['MACRO_F1'],
                'history': history,
                'final_metrics': final_metrics_complete
            }
            with open(run_results_path / 'training_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            pd.DataFrame([final_metrics_complete]).to_csv(evidence_path / 'final_metrics.csv', index=False)
            print(f"✅ Evidence saved to {run_results_path}")

        print("=== A/V CLASSIFICATION TRAINING COMPLETED ===\n")

    if args.train_od:
        print("\n=== STARTING OPTIC DISC DETECTION TRAINING ===")
        train_aug = SegmentationAugmentation(img_size=OPTIC_DISC_CONFIG["DATASET"]["IMAGE_SIZE"], phase="train").transform
        val_aug = SegmentationAugmentation(img_size=OPTIC_DISC_CONFIG["DATASET"]["IMAGE_SIZE"], phase="val").transform

        train_ds = OpticDiscDataset(
            OPTIC_DISC_CONFIG["DATASET"]["BASE_PATH"], phase="train",
            img_size=OPTIC_DISC_CONFIG["DATASET"]["IMAGE_SIZE"], transform=train_aug,
        )
        val_ds = OpticDiscDataset(
            OPTIC_DISC_CONFIG["DATASET"]["BASE_PATH"], phase="val",
            img_size=OPTIC_DISC_CONFIG["DATASET"]["IMAGE_SIZE"], transform=val_aug,
        )

        train_loader = DataLoader(train_ds, batch_size=OPTIC_DISC_CONFIG["TRAINING"]["BATCH_SIZE"], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=OPTIC_DISC_CONFIG["TRAINING"]["BATCH_SIZE"], shuffle=False, num_workers=0)

        model = EnhancedUNet(
            in_channels=OPTIC_DISC_CONFIG["MODEL"]["IN_CHANNELS"],
            out_channels=OPTIC_DISC_CONFIG["MODEL"]["OUT_CHANNELS"],
            features=OPTIC_DISC_CONFIG["MODEL"]["FEATURES"],
        )
        trainer = EnhancedSegmentationTrainer(model, train_loader, val_loader, resume=args.resume, config=OPTIC_DISC_CONFIG)

        best_model_path, history, final_metrics, run_id = trainer.train(epochs=OPTIC_DISC_CONFIG["TRAINING"]["EPOCHS"])

        if best_model_path and best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True))
            final_metrics = evaluate_best_model_od(model, val_loader)

            run_results_path = OPTIC_DISC_CONFIG["PATHS"]["RESULTS"] / run_id
            run_results_path.mkdir(parents=True, exist_ok=True)
            results = {
                'run_id': run_id,
                'best_dice_score': final_metrics['dice_score'],
                'target_achieved': final_metrics['dice_score'] >= OPTIC_DISC_CONFIG['TARGETS']['DICE_SCORE'],
                'history': history,
                'final_metrics': final_metrics
            }
            with open(run_results_path / 'training_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Optic disc model dice: {final_metrics['dice_score']:.4f} — evidence saved to {run_results_path}")

        print("=== OPTIC DISC DETECTION TRAINING COMPLETED ===\n")

    if args.run_pipeline:
        print("\n=== STARTING INTEGRATED PIPELINE ===")
        pipeline = ScientificAVRPipeline()
        if pipeline.load_models():
            results = pipeline.process_image(args.run_pipeline)
            if results and not results.get("error"):
                print("\n--- PIPELINE RESULTS ---")
                for k, v in results.items():
                    if isinstance(v, (float, np.float32)):
                        print(f"{k}: {v:.4f}")
                    elif not isinstance(v, (np.ndarray, torch.Tensor)):
                        print(f"{k}: {v}")
            else:
                print(f"***x*** Failed to process image: {results.get('error') if results else 'no result'}")
        else:
            print("***x*** Failed to load models for pipeline.")
        print("=== INTEGRATED PIPELINE COMPLETED ===\n")

if __name__ == "__main__":
    main()