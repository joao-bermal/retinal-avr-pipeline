import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config.settings import SEGMENTATION_CONFIG as C_SEG
from src.config.settings import AV_CLASSIFICATION_CONFIG as C_AV

# ====================================================================
# SEGMENTATION LOSSES
# ====================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class IoULoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum() - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou

class CombinedLossOptimized(nn.Module):
    def __init__(self):
        super(CombinedLossOptimized, self).__init__()
        
        self.dice_weight = C_SEG["LOSS"]["DICE_WEIGHT"]
        self.focal_weight = C_SEG["LOSS"]["FOCAL_WEIGHT"]
        self.bce_weight = C_SEG["LOSS"]["BCE_WEIGHT"]
        self.iou_weight = C_SEG["LOSS"]["IOU_WEIGHT"]
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        self.bce_loss = nn.BCELoss()
        self.iou_loss = IoULoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        if predictions.dim() == 4 and predictions.size(1) == 1:
            predictions = predictions.squeeze(1)
        
        dice_component = self.dice_loss(predictions, targets)
        focal_component = self.focal_loss(predictions, targets)
        bce_component = self.bce_loss(predictions, targets)
        iou_component = self.iou_loss(predictions, targets)
        
        total_loss = (
            self.dice_weight * dice_component +
            self.focal_weight * focal_component +
            self.bce_weight * bce_component +
            self.iou_weight * iou_component
        )
        
        loss_components = {
            'total_loss': total_loss.item(),
            'dice_loss': dice_component.item(),
            'focal_loss': focal_component.item(),
            'bce_loss': bce_component.item(),
            'iou_loss': iou_component.item()
        }
        
        return total_loss, loss_components

# ====================================================================
# AV CLASSIFICATION LOSSES
# ====================================================================

def dice_loss_multiclass(logits, targets, eps=1e-6):
    """
    Multiclass dice with softmax on logits; computed over all classes (0/1/2).
    Adjustment to ignore bg if the notebook SOTA does this (simply mask class 0).
    """
    num_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)                 # [B,C,H,W]
    targets_oh = torch.zeros_like(probs)                 # [B,C,H,W]
    targets_oh.scatter_(1, targets.unsqueeze(1), 1.0)
    dims = (0, 2, 3)
    intersect = torch.sum(probs * targets_oh, dim=dims)
    denom = torch.sum(probs, dim=dims) + torch.sum(targets_oh, dim=dims) + eps
    dice = (2.0 * intersect) / denom
    return 1.0 - dice.mean()

def focal_loss_multiclass(logits, targets, gamma=2.0, alpha=None):
    """
    Multiclass focal loss: ((1-pt)^gamma) * CE.
    alpha can be a vector of per-class weights (Tensor[C]).
    """
    ce = F.cross_entropy(logits, targets, reduction="none", weight=alpha)  # [B,H,W]
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    return focal.mean()

class SOTALoss(nn.Module):
    """
    Identical loss to SOTA notebook: total = BCE * w_bce + Dice * w_dice + Focal * w_focal.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        w = cfg["LOSS"]
        self.w_bce   = float(w.get("BCE_WEIGHT", 1.0))
        self.w_dice  = float(w.get("DICE_WEIGHT", 1.0))
        self.w_focal = float(w.get("FOCAL_WEIGHT", 0.0))  # Default to 0 if not present
        self.gamma   = float(w.get("FOCAL_GAMMA", 2.0))

        ds = cfg.get("DATASET", {})
        cw = ds.get("CLASS_WEIGHTS", None)
        self.class_weights = None
        if cw is not None:
            self.class_weights = torch.tensor(cw, dtype=torch.float32)

    def forward(self, logits, targets):
        # Safety: align HxW (in case model does not guarantee it in forward)
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        alpha = self.class_weights.to(logits.device) if self.class_weights is not None else None

        ce = F.cross_entropy(logits, targets, weight=alpha)
        dl = dice_loss_multiclass(logits, targets)
        fl = focal_loss_multiclass(logits, targets, gamma=self.gamma, alpha=alpha) if self.w_focal > 0 else torch.tensor(0.0, device=logits.device)

        total = self.w_bce * ce + self.w_dice * dl + self.w_focal * fl
        return {"total": total, "ce": ce, "dice": dl, "focal": fl}
