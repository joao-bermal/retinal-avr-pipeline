# src/metrics/av_metrics.py

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

@torch.no_grad()
def av_pixel_metrics(logits, targets):
    """
    Retorna dict {"macro_f1", "acc", "f1_art", "f1_vein"} no padrão SOTA.
    """
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy().astype(np.int32)
    y     = targets.cpu().numpy().astype(np.int32)
    yt, yp = y.reshape(-1), preds.reshape(-1)
    acc   = accuracy_score(yt, yp)
    macro = f1_score(yt, yp, average="macro")
    f1c   = f1_score(yt, yp, average=None, labels=[0,1,2])
    f1_art  = float(f1c[1]) if len(f1c) > 1 else 0.0
    f1_vein = float(f1c[2]) if len(f1c) > 2 else 0.0
    return {"macro_f1": float(macro), "acc": float(acc), "f1_art": f1_art, "f1_vein": f1_vein}
