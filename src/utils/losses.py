import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.settings import SEGMENTATION_CONFIG, AV_CLASSIFICATION_CONFIG

class CombinedLoss(nn.Module):
    """Função de perda combinada para segmentação (BCE + Dice)."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1e-6):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return 1 - dice

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice_loss(torch.sigmoid(inputs), targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss, {
            'bce_loss': bce_loss.item(),
            'dice_loss': dice_loss.item()
        }

class FocalLoss(nn.Module):
    """Focal Loss para desequilíbrio de classes"""
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            at = self.alpha[targets]
            focal_loss = at * focal_loss
        
        return focal_loss.mean()

class DiceLossAV(nn.Module):
    """Dice Loss para segmentação multi-classe (A/V) - renomeada para evitar conflito"""
    def __init__(self, smooth=1e-6):
        super(DiceLossAV, self).__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets, class_id):
        inputs_flat = torch.softmax(inputs, dim=1)[:, class_id].reshape(-1)
        targets_flat = (targets == class_id).float().reshape(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        return 1 - dice

class ChannelConsistencyLoss(nn.Module):
    """C3 Loss - Channel Consistency Loss"""
    def __init__(self):
        super(ChannelConsistencyLoss, self).__init__()
    
    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=1)
        
        # Calculate spatial gradients
        grad_x = torch.abs(probs[:, :, :-1, :] - probs[:, :, 1:, :])
        grad_y = torch.abs(probs[:, :, :, :-1] - probs[:, :, :, 1:])
        
        consistency_loss = grad_x.mean() + grad_y.mean()
        
        return consistency_loss

class IntraClassLoss(nn.Module):
    """Intra-class consistency loss"""
    def __init__(self):
        super(IntraClassLoss, self).__init__()
    
    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=1)
        
        intra_loss = 0.0
        num_classes = probs.shape[1]
        
        for c in range(1, num_classes):  # Skip background
            class_mask = (targets == c).float()
            if class_mask.sum() > 0:
                class_probs = probs[:, c] * class_mask
                class_mean = class_probs.sum() / (class_mask.sum() + 1e-6)
                class_var = ((class_probs - class_mean) ** 2 * class_mask).sum() / (class_mask.sum() + 1e-6)
                intra_loss += class_var
        
        return intra_loss / (num_classes - 1)

class EnhancedMultiLoss(nn.Module):
    """Enhanced Multi-Loss para Classificação A/V Multi-Dataset"""
    
    def __init__(self, config=AV_CLASSIFICATION_CONFIG):
        super(EnhancedMultiLoss, self).__init__()
        
        if 'CLASS_WEIGHTS' in config.get('LOSS', {}):
            class_weights = torch.tensor(config['LOSS']['CLASS_WEIGHTS']).float()
            alpha_values = config['LOSS']['CLASS_WEIGHTS']
        elif 'ALPHA' in config.get('LOSS', {}):
            alpha_values = config['LOSS']['ALPHA']
            class_weights = torch.tensor(alpha_values).float()
        else:
            alpha_values = [1.0, 5.0, 5.0]  # Background, Artery, Vein
            class_weights = torch.tensor(alpha_values).float()
            
        self.bce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss_av = DiceLossAV()
        self.focal_loss = FocalLoss(alpha=alpha_values, gamma=2.0)
        self.c3_loss = ChannelConsistencyLoss()
        self.intra_loss = IntraClassLoss()
        
        self.bce_weight = config['LOSS']['BCE_WEIGHT']
        self.dice_weight = config['LOSS']['DICE_WEIGHT']
        self.focal_weight = config['LOSS']['FOCAL_WEIGHT']
        self.c3_weight = config['LOSS']['C3_WEIGHT']
        self.intra_weight = config['LOSS']['INTRA_WEIGHT']
        
    def forward(self, inputs, targets):
        # BCE Loss
        bce = self.bce_loss(inputs, targets)
        
        # Focal Loss
        focal = self.focal_loss(inputs, targets)
        
        # Dice losses para classes A/V
        try:
            dice_artery = self.dice_loss_av(inputs, targets, class_id=1)
            dice_vein = self.dice_loss_av(inputs, targets, class_id=2)
            dice_total = (dice_artery + dice_vein) / 2
        except:
            dice_total = torch.tensor(0.0, device=inputs.device)
        
        # C3 Loss
        c3 = self.c3_loss(inputs, targets)
        
        # Intra-class Loss
        intra = self.intra_loss(inputs, targets)
        
        total_loss = (
            self.bce_weight * bce +
            self.dice_weight * dice_total +
            self.focal_weight * focal +
            self.c3_weight * c3 +
            self.intra_weight * intra
        )
        
        return {
            'total': total_loss,
            'bce': bce.item(),
            'dice_av': dice_total.item(),
            'focal': focal.item(),
            'c3': c3.item(),
            'intra': intra.item()
        }