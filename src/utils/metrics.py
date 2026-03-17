# -*- coding: utf-8 -*-

import torch
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from typing import Dict, List
import matplotlib.pyplot as plt
from pathlib import Path

from src.config.settings import SEGMENTATION_CONFIG, AV_CLASSIFICATION_CONFIG

def calculate_comprehensive_metrics_segmentation(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """Calcula métricas abrangentes para segmentação (Dice, IoU, Accuracy, Sensitivity, Specificity, Precision, F1, AUC-ROC)."""
    
    # Certificar que os tensores estão na CPU e são numpy arrays
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()

    # Binarizar predições e targets para métricas baseadas em limiar
    pred_binary = (pred_np > 0.5).astype(np.uint8)
    targets_binary = (target_np > 0.5).astype(np.uint8)

    # Achatamento para sklearn
    pred_flat = pred_binary.flatten()
    targets_flat = targets_binary.flatten()
    pred_prob_flat = pred_np.flatten()

    # === DICE SCORE ===
    intersection = (pred_flat * targets_flat).sum()
    dice = (2. * intersection + 1e-6) / (pred_flat.sum() + targets_flat.sum() + 1e-6)

    # === ACCURACY ===
    accuracy = accuracy_score(targets_flat, pred_flat)

    # === PRECISION, RECALL, F1-SCORE ===
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_flat, pred_flat, average='binary', zero_division=0
    )

    # === IoU / JACCARD INDEX ===
    union = pred_flat.sum() + targets_flat.sum() - intersection
    iou = intersection / (union + 1e-6)
    
    # === ESPECIFICIDADE (True Negative Rate) ===
    tn = ((1 - pred_binary) * (1 - targets_binary)).sum()
    fp = (pred_binary * (1 - targets_binary)).sum()
    specificity = tn / (tn + fp + 1e-6)
    
    # === AUC-ROC ===
    try:
        unique_targets = np.unique(targets_flat)
        auc = roc_auc_score(targets_flat, pred_prob_flat) if len(unique_targets) > 1 else 0.0
    except Exception:
        auc = 0.0
    
    # === MONTAGEM DO DICIONÁRIO DE MÉTRICAS ===
    metrics = {
        'dice_score': float(dice),
        'iou': float(iou),
        'accuracy': float(accuracy),
        'sensitivity': float(recall),  # Recall = Sensitivity
        'specificity': float(specificity),
        'precision': float(precision),
        'f1_score': float(f1),
        'auc_roc': float(auc)
    }
    
    return metrics

def print_metrics_comparison_segmentation(metrics: Dict[str, float], 
                                          targets: Dict[str, float] = None):
    """Imprime métricas formatadas com comparação aos targets para segmentação."""
    
    if targets is None:
        targets = {
            'dice_score': SEGMENTATION_CONFIG['TARGETS']['DICE_SCORE'],
            'accuracy': SEGMENTATION_CONFIG['TARGETS']['ACCURACY'],
            'sensitivity': SEGMENTATION_CONFIG['TARGETS']['SENSITIVITY'],
            'specificity': SEGMENTATION_CONFIG['TARGETS']['SPECIFICITY'],
            'iou': SEGMENTATION_CONFIG['TARGETS']['IOU']
        }
    
    print("\n📊 MÉTRICAS DE PERFORMANCE (Segmentação):")
    print("-" * 50)
    
    for metric_name, value in metrics.items():
        target_value = targets.get(metric_name)
        
        if target_value:
            status = "***" if value >= target_value else "⚠️"
            print(f"{status} {metric_name.title():15}: {value:.4f} (meta: {target_value:.4f})")
        else:
            print(f"📈 {metric_name.title():15}: {value:.4f}")

def create_metrics_visualization_segmentation(history: Dict, save_path: Path = None):
    """Cria visualização das métricas durante o treinamento de segmentação."""
    
    epochs = range(1, len(history['train_dice']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Dice Score
    axes[0, 0].plot(epochs, history['train_dice'], 'b-', label='Treino', linewidth=2)
    axes[0, 0].plot(epochs, history['val_dice'], 'r-', label='Validação', linewidth=2)
    axes[0, 0].axhline(y=SEGMENTATION_CONFIG['TARGETS']['DICE_SCORE'], color='g', linestyle='--', 
                      label=f'Meta ({SEGMENTATION_CONFIG["TARGETS"]["DICE_SCORE"]})', alpha=0.7)
    axes[0, 0].set_title('Dice Score')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Dice Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(epochs, history['train_loss'], 'b-', label='Treino', linewidth=2)
    axes[0, 1].plot(epochs, history['val_loss'], 'r-', label='Validação', linewidth=2)
    axes[0, 1].set_title('Combined Loss')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # IoU
    axes[1, 0].plot(epochs, history['train_iou'], 'b-', label='Treino', linewidth=2)
    axes[1, 0].plot(epochs, history['val_iou'], 'r-', label='Validação', linewidth=2)
    axes[1, 0].axhline(y=SEGMENTATION_CONFIG['TARGETS']['IOU'], color='g', linestyle='--', 
                      label=f'Meta ({SEGMENTATION_CONFIG["TARGETS"]["IOU"]})', alpha=0.7)
    axes[1, 0].set_title('IoU Score')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate (se disponível)
    if 'learning_rates' in history:
        axes[1, 1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
    
    plt.suptitle('Enhanced U-Net - Evolução das Métricas', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def create_comprehensive_training_analysis_segmentation(history: Dict, 
                                                      final_metrics: Dict,
                                                      save_path: Path = None):
    """Cria análise completa do treinamento como na fig03_training_curves_complete.jpg"""
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Evolução da Loss Function
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    # Assuming best epoch is known or can be derived from history
    best_epoch_idx = np.argmax(history['val_dice'])
    best_epoch = epochs[best_epoch_idx]
    axes[0, 0].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7, label=f'Melhor Época ({best_epoch})')
    axes[0, 0].set_title('Evolução da Loss Function', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Combined Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Evolução do Dice Score
    axes[0, 1].plot(epochs, history['train_dice'], 'b-', label='Training Dice', linewidth=2)
    axes[0, 1].plot(epochs, history['val_dice'], 'r-', label='Validation Dice', linewidth=2)
    axes[0, 1].axhline(y=SEGMENTATION_CONFIG['TARGETS']['DICE_SCORE'], color='orange', linestyle=':', alpha=0.8, 
                      label=f'Meta TCC ({SEGMENTATION_CONFIG["TARGETS"]["DICE_SCORE"]})')
    axes[0, 1].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7, 
                      label=f'Melhor Época ({best_epoch})')
    axes[0, 1].axhline(y=final_metrics['dice_score'], color='green', linestyle='--', alpha=0.8, 
                      label=f'Melhor Resultado ({final_metrics["dice_score"]:.4f})')
    axes[0, 1].set_title('Evolução do Dice Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Evolução do IoU
    axes[1, 0].plot(epochs, history['train_iou'], 'b-', label='Training IoU', linewidth=2)
    axes[1, 0].plot(epochs, history['val_iou'], 'r-', label='Validation IoU', linewidth=2)
    axes[1, 0].axhline(y=SEGMENTATION_CONFIG['TARGETS']['IOU'], color='orange', linestyle=':', alpha=0.8, 
                      label=f'Meta IoU ({SEGMENTATION_CONFIG["TARGETS"]["IOU"]})')
    axes[1, 0].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7, 
                      label=f'Melhor Época ({best_epoch})')
    axes[1, 0].set_title('Evolução do IoU', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('IoU Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Learning Rate Schedule
    if 'learning_rates' in history:
        axes[1, 1].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule (ReduceLROnPlateau)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Época')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Análise de Overfitting
    train_val_diff = np.array(history['val_dice']) - np.array(history['train_dice'])
    axes[0, 2].plot(epochs, train_val_diff, 'purple', linewidth=2)
    axes[0, 2].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, 
                      label='Threshold Overfitting (0.05)')
    axes[0, 2].set_title('Análise de Overfitting', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Época')
    axes[0, 2].set_ylabel('Train - Val Dice')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 6. Resumo Estatístico Completo (adaptado)
    summary_text = f"""RESUMO DO TREINAMENTO:\n\n"
    f"RESULTADO FINAL:\n"
    f"• Melhor Dice Score: {final_metrics['dice_score']:.4f}\n"
    f"• Melhor Época: {best_epoch}/{len(epochs)}\n"
    f"• Meta TCC: {SEGMENTATION_CONFIG['TARGETS']['DICE_SCORE']} {'✓ ATINGIDA' if final_metrics['dice_score'] >= SEGMENTATION_CONFIG['TARGETS']['DICE_SCORE'] else '✗ NÃO ATINGIDA'}\n"
    f"• Status: {'SUCESSO COMPLETO' if final_metrics['dice_score'] >= SEGMENTATION_CONFIG['TARGETS']['DICE_SCORE'] else 'PROGRESSO'}\n\n"
    f"PERFORMANCE FINAL:\n"
    f"• Training Dice: {history['train_dice'][-1]:.4f}\n"
    f"• Validation Dice: {history['val_dice'][-1]:.4f}\n"
    f"• Gap Train/Val: {abs(history['train_dice'][-1] - history['val_dice'][-1]):.4f}\n"
    f"• IoU final: {final_metrics['iou']:.4f}\n"
    f"• Loss final: {final_metrics['total_loss']:.4f}\n\n"
    f"CONFIGURAÇÕES ÓTIMAS:\n"
    f"✓ Combined Loss Function\n"
    f"✓ Learning Rate: {SEGMENTATION_CONFIG['TRAINING']['LEARNING_RATE']}\n"
    f"✓ Batch Size: {SEGMENTATION_CONFIG['TRAINING']['BATCH_SIZE']}\n"
    f"✓ Dropout: 0.5 (se aplicável)\n"
    f"✓ Early Stopping: {SEGMENTATION_CONFIG['TRAINING']['EARLY_STOPPING_PATIENCE']}\n"
    f"""    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes,
                   fontsize=9, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    axes[1, 2].set_title('Resumo Estatístico Completo', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle('Enhanced U-Net: Análise Completa do Processo de Treinamento', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def create_architecture_analysis_complete_segmentation(model: torch.nn.Module, save_path: Path = None):
    """Cria análise arquitetural completa como na fig02_architecture_analysis.jpg"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. Enhanced U-Net Architecture Diagram (Simplificado)
    ax1.text(0.5, 0.9, 'Enhanced U-Net Architecture', ha='center', fontsize=16, fontweight='bold')
    
    # Desenhar diagrama da arquitetura (representação abstrata)
    levels = SEGMENTATION_CONFIG['MODEL']['FEATURES'] + [SEGMENTATION_CONFIG['MODEL']['FEATURES'][-1] * 2] + list(reversed(SEGMENTATION_CONFIG['MODEL']['FEATURES']))
    positions = np.linspace(0.1, 0.9, len(levels))
    colors = ['lightcoral'] * len(SEGMENTATION_CONFIG['MODEL']['FEATURES']) + ['red'] + ['lightgreen'] * len(SEGMENTATION_CONFIG['MODEL']['FEATURES'])
    
    for i, (pos, level, color) in enumerate(zip(positions, levels, colors)):
        rect = plt.Rectangle((pos-0.04, 0.4), 0.08, 0.3, 
                           facecolor=color, edgecolor='black', linewidth=1)
        ax1.add_patch(rect)
        ax1.text(pos, 0.55, str(level), ha='center', va='center', fontweight='bold')
        
        if i < len(levels)-1:
            arrow_style = '→' if i < len(SEGMENTATION_CONFIG['MODEL']['FEATURES']) else '←'
            ax1.text(pos+0.06, 0.55, arrow_style, ha='center', va='center', fontsize=12)
    
    total_params = sum(p.numel() for p in model.parameters())
    ax1.text(0.5, 0.2, f'Input: {SEGMENTATION_CONFIG["DATASET"]["IMAGE_SIZE"][0]}x{SEGMENTATION_CONFIG["DATASET"]["IMAGE_SIZE"][1]}x3\nOutput: {SEGMENTATION_CONFIG["DATASET"]["IMAGE_SIZE"][0]}x{SEGMENTATION_CONFIG["DATASET"]["IMAGE_SIZE"][1]}x1\nParâmetros: {total_params:,}', 
             ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightyellow'))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # 2. Distribuição de Parâmetros por Componente (Estimativa)
    # Estes valores são aproximados e devem ser calculados com base na arquitetura real se necessário
    components = ['Encoder', 'Bottleneck', 'Decoder', 'Final Conv']
    # Exemplo de distribuição, ajuste conforme a arquitetura real do modelo
    params_dist = [total_params * 0.3, total_params * 0.4, total_params * 0.25, total_params * 0.05]
    
    ax2.bar(range(len(components)), params_dist, color='skyblue', alpha=0.7)
    ax2.set_title('Distribuição de Parâmetros por Componente', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Componentes da Arquitetura')
    ax2.set_ylabel('Número de Parâmetros')
    ax2.set_xticks(range(len(components)))
    ax2.set_xticklabels(components, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Curva ROC (Exemplo, pois requer dados de teste)
    ax3.text(0.5, 0.5, 'Curva ROC\n(Requer dados de teste para geração)', 
             ha='center', va='center', transform=ax3.transAxes,
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax3.set_title('Curva ROC', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 4. Curva Precision-Recall (Exemplo, pois requer dados de teste)
    ax4.text(0.5, 0.5, 'Curva Precision-Recall\n(Requer dados de teste para geração)', 
             ha='center', va='center', transform=ax4.transAxes,
             fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax4.set_title('Curva Precision-Recall', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle('Enhanced U-Net: Análise Arquitetural e de Performance', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def create_sample_predictions_plot_segmentation(model: torch.nn.Module, 
                                                  dataset: torch.utils.data.Dataset, 
                                                  device: torch.device,
                                                  num_samples: int = 3,
                                                  save_path: Path = None):
    """Cria comparação visual Original vs GT vs Predição para segmentação."""
    
    model.eval()
    
    # Selecionar amostras aleatórias
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Carregar amostra
            image, gt_mask = dataset[idx]
            
            # Desnormalizar imagem para visualização
            if isinstance(image, torch.Tensor):
                if image.shape[0] == 3:  # RGB
                    # Desnormalizar ImageNet
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image_vis = image * std + mean
                    image_vis = torch.clamp(image_vis, 0, 1)
                    image_vis = image_vis.permute(1, 2, 0).numpy()
                else:
                    image_vis = image.numpy()
            else:
                image_vis = image
            
            # Fazer predição
            input_image = image.unsqueeze(0).to(device)
            pred_mask = model(input_image)
            pred_mask = (pred_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            
            # Ground truth
            if isinstance(gt_mask, torch.Tensor):
                gt_mask = gt_mask.squeeze().numpy()
            
            # Plotar
            # Original
            axes[i, 0].imshow(image_vis)
            axes[i, 0].set_title(f'Sample {idx}: Original Image')
            axes[i, 0].axis('off')
            
            # Ground Truth
            axes[i, 1].imshow(gt_mask, cmap='gray')
            axes[i, 1].set_title(f'Sample {idx}: Ground Truth')
            axes[i, 1].axis('off')
            
            # Predição
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title(f'Sample {idx}: Prediction')
            axes[i, 2].axis('off')
    
    plt.suptitle('Enhanced U-Net - Sample Predictions\nRetinal Vessel Segmentation', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def create_metrics_boxplot_segmentation(metrics_per_image: List[Dict[str, float]], 
                                        save_path: Path = None):
    """Cria boxplot de distribuição de métricas por imagem para segmentação."""
    
    # Converter para DataFrame
    import pandas as pd
    df = pd.DataFrame(metrics_per_image)
    
    # Selecionar métricas principais
    metrics_to_plot = ['dice_score', 'iou', 'accuracy', 'sensitivity', 'specificity']
    df_plot = df[metrics_to_plot]
    
    # Plotar
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    box_plot = ax.boxplot([df_plot[col].values for col in metrics_to_plot], 
                         labels=[col.replace('_', ' ').title() for col in metrics_to_plot],
                         patch_artist=True, notch=True)
    
    # Colorir boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightgray']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title('Distribution of Metrics Across Test Images\nEnhanced U-Net - Retinal Vessel Segmentation', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def create_confusion_matrix_plot_av(y_true, y_pred, class_names=['Background', 'Artery', 'Vein'], save_path: Path = None):
    """Cria e salva a matriz de confusão para classificação A/V."""
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=np.arange(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title('Confusion Matrix - Enhanced Multi-Dataset AV-Net')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_precision_recall_curve_av(y_true_binary, y_scores_clipped, save_path: Path = None):
    """Cria e salva a curva Precision-Recall para classificação A/V."""
    from sklearn.metrics import precision_recall_curve, auc
    
    try:
        precision, recall, _ = precision_recall_curve(y_true_binary, y_scores_clipped)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
        ax.fill_between(recall, precision, alpha=0.2, color='blue')
        
        ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve - Enhanced Multi-Dataset AV-Net\nArtery/Vein Classification', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)
        
    except Exception as e:
        print(f"***x*** Erro ao criar curva Precision-Recall: {e}")
        print("📊 Gerando plot alternativo...")
        
        # Plot alternativo simples
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, f'Precision-Recall Curve\n(Erro na geração)\n\nDados disponíveis:\n• Amostras: {len(y_true_binary)}\n• Scores range: [{y_scores_clipped.min():.3f}, {y_scores_clipped.max():.3f}]', 
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax.set_title('Precision-Recall Curve - Enhanced Multi-Dataset AV-Net', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)

def create_sample_predictions_plot_av(model: torch.nn.Module, 
                                      dataset: torch.utils.data.Dataset, 
                                      device: torch.device,
                                      num_samples: int = 3,
                                      save_path: Path = None):
    """Cria comparação visual Original vs GT vs Predição para classificação A/V."""
    
    model.eval()
    
    # Selecionar amostras aleatórias
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Colormap para A/V (como na imagem científica)
    from matplotlib.colors import ListedColormap
    colors = ['#000000', '#FF0000', '#0080FF']  # Black, Red, Blue
    av_cmap = ListedColormap(colors)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Carregar amostra
            sample = dataset[idx]
            image = sample['image']
            gt_mask = sample['mask']
            
            # Desnormalizar imagem para visualização
            if isinstance(image, torch.Tensor):
                if image.shape[0] == 3:  # RGB
                    # Desnormalizar ImageNet
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    image_vis = image * std + mean
                    image_vis = torch.clamp(image_vis, 0, 1)
                    image_vis = image_vis.permute(1, 2, 0).numpy()
                else:
                    image_vis = image.numpy()
            else:
                image_vis = image
            
            # Fazer predição
            input_image = image.unsqueeze(0).to(device)
            pred_output = model(input_image)
            pred_mask = torch.argmax(pred_output, dim=1).squeeze().cpu().numpy()
            
            # Ground truth
            if isinstance(gt_mask, torch.Tensor):
                gt_mask = gt_mask.squeeze().numpy()
            
            # Plotar
            # Original
            axes[i, 0].imshow(image_vis)
            axes[i, 0].set_title(f'Sample {idx}: Original Image')
            axes[i, 0].axis('off')
            
            # Ground Truth
            axes[i, 1].imshow(gt_mask, cmap=av_cmap, vmin=0, vmax=2)
            axes[i, 1].set_title(f'Sample {idx}: Ground Truth')
            axes[i, 1].axis('off')
            
            # Predição
            axes[i, 2].imshow(pred_mask, cmap=av_cmap, vmin=0, vmax=2)
            axes[i, 2].set_title(f'Sample {idx}: Prediction')
            axes[i, 2].axis('off')
    
    plt.suptitle('Enhanced Multi-Dataset AV-Net - Sample Predictions\nArtery/Vein Classification', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def create_final_consolidated_analysis_av(final_metrics: Dict, save_path: Path = None):
    """Cria uma análise consolidada final para classificação A/V."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    summary_text = f"""ANÁLISE CONSOLIDADA FINAL (A/V Classification):\n\n"
    f"RESULTADOS GERAIS:\n"
    f"• Macro F1-Score: {final_metrics.get('f1', 0.0):.4f}\n"
    f"• Accuracy: {final_metrics.get('accuracy', 0.0):.4f}\n\n"
    f"F1-SCORE POR CLASSE:\n"
    f"• Background F1: {final_metrics.get('bg_f1', 0.0):.4f}\n"
    f"• Artery F1: {final_metrics.get('artery_f1', 0.0):.4f}\n"
    f"• Vein F1: {final_metrics.get('vein_f1', 0.0):.4f}\n\n"
    f"METAS TCC (AV Classification):\n"
    f"• Macro F1: {AV_CLASSIFICATION_CONFIG['TARGETS']['MACRO_F1']} {'✓ ATINGIDA' if final_metrics.get('f1', 0.0) >= AV_CLASSIFICATION_CONFIG['TARGETS']['MACRO_F1'] else '✗ NÃO ATINGIDA'}\n"
    f"• Accuracy: {AV_CLASSIFICATION_CONFIG['TARGETS']['ACCURACY']} {'✓ ATINGIDA' if final_metrics.get('accuracy', 0.0) >= AV_CLASSIFICATION_CONFIG['TARGETS']['ACCURACY'] else '✗ NÃO ATINGIDA'}\n"
    f"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    ax.set_title('Análise Consolidada de Classificação A/V', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)

def create_professional_comparison_av(val_predictions, val_targets, num_samples=6, save_path: Path = None):
    """Criar visualização profissional estilo paper científico para classificação A/V."""
    
    # Configurar estilo científico
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'sans-serif',
        'axes.titlesize': 11,
        'axes.labelsize': 10
    })
    
    # Colormap para A/V (como na imagem científica)
    from matplotlib.colors import ListedColormap
    colors = ['#000000', '#FF0000', '#0080FF']  # Black, Red, Blue
    av_cmap = ListedColormap(colors)
    
    # Selecionar amostras
    sample_indices = list(range(min(num_samples, len(val_predictions))))
    
    # Criar figura com grid profissional
    fig = plt.figure(figsize=(16, 4 * len(sample_indices)))
    
    # Título principal
    fig.suptitle('Enhanced A/V Classification - Test Results\nMulti-Dataset Model', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    for i, idx in enumerate(sample_indices):
        
        # Get data
        ground_truth = val_targets[idx]
        prediction = val_predictions[idx]
        
        # Calcular erro (diferenças pixel-wise)
        error_map = np.abs(ground_truth - prediction)
        
        # Create subplot row for this sample
        row_start = i * 4
        
        # Sample info
        ax1 = plt.subplot(len(sample_indices), 4, row_start + 1)
        ax1.text(0.5, 0.5, f'Sample {idx+1}\nMulti-Dataset\nValidation', 
                ha='center', va='center', fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title('Sample Info', fontweight='bold')
        ax1.axis('off')
        
        # Ground Truth
        ax2 = plt.subplot(len(sample_indices), 4, row_start + 2)
        ax2.imshow(ground_truth, cmap=av_cmap, vmin=0, vmax=2)
        ax2.set_title('Ground Truth', fontweight='bold')
        ax2.axis('off')
        
        # Prediction
        ax3 = plt.subplot(len(sample_indices), 4, row_start + 3)
        ax3.imshow(prediction, cmap=av_cmap, vmin=0, vmax=2)
        ax3.set_title('Prediction', fontweight='bold')
        ax3.axis('off')
        
        # Error Map
        ax4 = plt.subplot(len(sample_indices), 4, row_start + 4)
        ax4.imshow(error_map, cmap='hot', vmin=0, vmax=2) # 0=correct, 1=misclassified as other vessel, 2=misclassified as background
        ax4.set_title('Error Map', fontweight='bold')
        ax4.axis('off')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)