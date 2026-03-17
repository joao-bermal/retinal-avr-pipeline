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
from src.config.settings import SEGMENTATION_CONFIG, AV_CLASSIFICATION_CONFIG, PIPELINE_CONFIG
from src.data.segmentation_dataset import EnhancedDRIVEDataset, SegmentationAugmentation
from src.data.av_classification_dataset import EnhancedIOSTARDataset, MultiDatasetAVClassification
from src.models.segmentation_model import EnhancedUNet
from src.models.av_classification_model import EnhancedMultiDatasetAVNet

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
    parser = argparse.ArgumentParser(description="Pipeline de Análise de Retinografia para TCC")
    parser.add_argument("--train_seg", action="store_true", help="Treinar o modelo de segmentação de vasos.")
    parser.add_argument("--train_av", action="store_true", help="Treinar o modelo de classificação A/V.")
    parser.add_argument("--run_pipeline", type=str, help="Executar o pipeline integrado em uma imagem específica.")
    parser.add_argument("--resume", action="store_true", help="Continuar treinamento de um checkpoint existente.")
    parser.add_argument("--config", type=str, help="Caminho para um arquivo .yaml para sobrescrever as configurações padrões do settings.py.")
    parser.add_argument("--lr", type=float, help="Sobrescrever a Learning Rate.")
    parser.add_argument("--epochs", type=int, help="Sobrescrever o número de Epochs.")
    parser.add_argument("--batch_size", type=int, help="Sobrescrever o Batch Size.")
    
    args = parser.parse_args()

    # Apply configuration overrides
    if args.config:
        update_config_from_yaml(SEGMENTATION_CONFIG, args.config)
        update_config_from_yaml(AV_CLASSIFICATION_CONFIG, args.config)
    
    # Direct overrides
    if args.lr:
        SEGMENTATION_CONFIG["TRAINING"]["LEARNING_RATE"] = args.lr
        AV_CLASSIFICATION_CONFIG["TRAINING"]["LEARNING_RATE"] = args.lr
    if args.epochs:
        SEGMENTATION_CONFIG["TRAINING"]["EPOCHS"] = args.epochs
        AV_CLASSIFICATION_CONFIG["TRAINING"]["EPOCHS"] = args.epochs
    if args.batch_size:
        SEGMENTATION_CONFIG["TRAINING"]["BATCH_SIZE"] = args.batch_size
        AV_CLASSIFICATION_CONFIG["TRAINING"]["BATCH_SIZE"] = args.batch_size

    # Garantir que os diretórios de saída existam
    for path_config in [SEGMENTATION_CONFIG["PATHS"], AV_CLASSIFICATION_CONFIG["PATHS"], {"OUTPUT_DIR": PIPELINE_CONFIG["OUTPUT_DIR"]}]:
        for key, path_obj in path_config.items():
            if isinstance(path_obj, Path):
                path_obj.mkdir(parents=True, exist_ok=True)

    if args.train_seg:
        print("\n=== INICIANDO TREINAMENTO DE SEGMENTAÇÃO DE VASOS ===")
        # Build transforms and loaders
        train_aug = SegmentationAugmentation(img_size=SEGMENTATION_CONFIG["DATASET"]["IMAGE_SIZE"], phase="train")
        val_aug = SegmentationAugmentation(img_size=SEGMENTATION_CONFIG["DATASET"]["IMAGE_SIZE"], phase="val")
        
        train_ds = EnhancedDRIVEDataset(SEGMENTATION_CONFIG["DATASET"]["BASE_PATH"], phase="train", transform=train_aug.transform)
        val_ds = EnhancedDRIVEDataset(SEGMENTATION_CONFIG["DATASET"]["BASE_PATH"], phase="test", transform=val_aug.transform)
        
        train_loader = DataLoader(train_ds, batch_size=SEGMENTATION_CONFIG["TRAINING"]["BATCH_SIZE"], shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=SEGMENTATION_CONFIG["TRAINING"]["BATCH_SIZE"], shuffle=False, num_workers=4, pin_memory=True)
        
        model = EnhancedUNet(in_channels=SEGMENTATION_CONFIG["MODEL"]["IN_CHANNELS"], out_channels=SEGMENTATION_CONFIG["MODEL"]["OUT_CHANNELS"])
        trainer = EnhancedSegmentationTrainer(model, train_loader, val_loader)
        trainer.train(epochs=SEGMENTATION_CONFIG["TRAINING"]["EPOCHS"])
        print("=== TREINAMENTO DE SEGMENTAÇÃO CONCLUÍDO ===\n")
    
    if args.train_av:
        print("\n=== INICIANDO TREINAMENTO DE CLASSIFICAÇÃO A/V ===")
        train_ds = MultiDatasetAVClassification(AV_CLASSIFICATION_CONFIG, phase='train')
        val_ds = MultiDatasetAVClassification(AV_CLASSIFICATION_CONFIG, phase='val')
        
        train_loader = DataLoader(train_ds, batch_size=AV_CLASSIFICATION_CONFIG["TRAINING"]["BATCH_SIZE"], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=AV_CLASSIFICATION_CONFIG["TRAINING"]["BATCH_SIZE"], shuffle=False, num_workers=4)
        
        model = EnhancedMultiDatasetAVNet(AV_CLASSIFICATION_CONFIG).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        trainer = EnhancedMultiDatasetTrainer(model, AV_CLASSIFICATION_CONFIG, torch.device("cuda" if torch.cuda.is_available() else "cpu"), AV_CLASSIFICATION_CONFIG["PATHS"]["LOGS"])
        
        trainer.fit(train_loader, val_loader, "cli_run", AV_CLASSIFICATION_CONFIG["PATHS"]["MODELS"])
        print("=== TREINAMENTO DE CLASSIFICAÇÃO A/V CONCLUÍDO ===\n")
        
    if args.run_pipeline:
        print("\n=== INICIANDO PIPELINE INTEGRADO ===")
        pipeline = ScientificAVRPipeline()
        if pipeline.load_models():
            # mock process_image since it's just a ref to an undefined method in the pipeline for now. Usually handled inside standard notebooks
            if hasattr(pipeline, "process_image"):
                results = pipeline.process_image(args.run_pipeline)
                if results:
                    print("\n--- RESULTADOS DO PIPELINE ---")
                    for k, v in results.items():
                        if isinstance(v, (float, np.float32)):
                            print(f"{k}: {v:.4f}")
                        elif not isinstance(v, (np.ndarray, torch.Tensor)):
                            print(f"{k}: {v}")
                else:
                    print("***x*** Falha ao processar a imagem.")
        else:
            print("***x*** Falha ao carregar os modelos para o pipeline.")
        print("=== PIPELINE INTEGRADO CONCLUÍDO ===\n")

if __name__ == "__main__":
    main()