import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config.settings import SEGMENTATION_CONFIG as C_SEG
from src.config.settings import AV_CLASSIFICATION_CONFIG as C_AV

# ====================================================================
# SEGMENTATION LOSSES
# ====================================================================

def dice_loss(probs, target, eps=1e-6):
    probs = probs.clamp(0,1)
    num = 2*(probs*target).sum()
    den = probs.sum() + target.sum() + eps
    return 1 - (num/den)

def iou_loss(probs, target, eps=1e-6):
    probs = probs.clamp(0,1)
    inter = (probs*target).sum()
    union = probs.sum() + target.sum() - inter + eps
    return 1 - inter/union

class FocalLossProb(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
    def forward(self, probs, targets):
        probs = probs.clamp(1e-6, 1-1e-6)
        bce = -(targets*torch.log(probs) + (1-targets)*torch.log(1-probs))
        pt  = torch.exp(-bce)
        return (self.alpha * (1-pt)**self.gamma * bce).mean()

class CombinedLossOptimized(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLossProb(gamma=2.0, alpha=0.25)
        self.bce   = nn.BCELoss()         
    def forward(self, preds, targets):
        probs = preds                        
        dl = dice_loss(probs, targets)
        il = iou_loss(probs, targets)
        fl = self.focal(probs, targets)
        bl = self.bce(probs, targets)
        return (C_SEG["LOSS"]["DICE_WEIGHT"]*dl + C_SEG["LOSS"]["FOCAL_WEIGHT"]*fl + C_SEG["LOSS"]["BCE_WEIGHT"]*bl + C_SEG["LOSS"]["IOU_WEIGHT"]*il)

# ====================================================================
# AV CLASSIFICATION LOSSES
# ====================================================================

def dice_loss_multiclass(logits, targets, eps=1e-6):
    """
    Dice multiclasse com softmax nos logits; calcula sobre todas as classes (0/1/2).
    Ajuste para ignorar bg se o SOTA do caderno fizer isso (basta mascarar a classe 0).
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
    Focal loss multiclasse: ((1-pt)^gamma) * CE.
    alpha pode ser vetor de pesos por classe (Tensor[C]).
    """
    ce = F.cross_entropy(logits, targets, reduction="none", weight=alpha)  # [B,H,W]
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    return focal.mean()

class SOTALoss(nn.Module):
    """
    Loss idêntica ao notebook SOTA: total = BCE * w_bce + Dice * w_dice + Focal * w_focal.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        w = cfg["LOSS"]
        self.w_bce   = float(w.get("BCE_WEIGHT", 1.0))
        self.w_dice  = float(w.get("DICE_WEIGHT", 1.0))
        self.w_focal = float(w.get("FOCAL_WEIGHT", 1.0))
        self.gamma   = float(w.get("FOCAL_GAMMA", 2.0))

        ds = cfg.get("DATASET", {})
        cw = ds.get("CLASS_WEIGHTS", None)
        self.class_weights = None
        if cw is not None:
            self.class_weights = torch.tensor(cw, dtype=torch.float32)

    def forward(self, logits, targets):
        # Segurança: alinhar HxW (caso o modelo não garanta no forward)
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        alpha = self.class_weights.to(logits.device) if self.class_weights is not None else None

        ce = F.cross_entropy(logits, targets, weight=alpha)
        dl = dice_loss_multiclass(logits, targets)
        fl = focal_loss_multiclass(logits, targets, gamma=self.gamma, alpha=alpha) if self.w_focal > 0 else torch.tensor(0.0, device=logits.device)

        total = self.w_bce * ce + self.w_dice * dl + self.w_focal * fl
        return {"total": total, "ce": ce, "dice": dl, "focal": fl}
