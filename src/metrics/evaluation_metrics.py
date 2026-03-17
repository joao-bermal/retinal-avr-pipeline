import torch

@torch.no_grad()
def compute_segmentation_metrics(pred, target, thr=0.5, eps=1e-6):
    binp = (pred > thr).float()
    tp = (binp*target).sum()
    fp = (binp*(1-target)).sum()
    fn = ((1-binp)*target).sum()
    tn = (((1-binp)*(1-target))).sum()
    dice = (2*tp)/(2*tp+fp+fn+eps)
    acc  = (tp+tn)/(tp+tn+fp+fn+eps)
    sen  = tp/(tp+fn+eps)
    spe  = tn/(tn+fp+eps)
    iou  = tp/(tp+fp+fn+eps)
    return dice.item(), acc.item(), sen.item(), spe.item(), iou.item()
