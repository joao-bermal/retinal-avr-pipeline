# src/trainers/av_trainer.py

import math
import csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path

from src.training.losses import SOTALoss
from src.metrics.av_metrics import av_pixel_metrics

class EnhancedMultiDatasetTrainer:
    def __init__(self, model, cfg, device, log_dir: Path):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.log_dir = Path(log_dir); self.log_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = SOTALoss(cfg)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["TRAINING"]["LEARNING_RATE"],
            weight_decay=cfg["TRAINING"].get("WEIGHT_DECAY", 1e-4),
        )
        self.epochs = cfg["TRAINING"]["EPOCHS"]
        self.warmup = cfg["TRAINING"].get("WARMUP_EPOCHS", 5)
        self.early  = cfg["TRAINING"].get("EARLY_STOPPING_PATIENCE", 25)

        def lr_lambda(ep):
            if ep < self.warmup:
                return (ep + 1)/max(1, self.warmup)
            t = ep - self.warmup
            T = max(1, self.epochs - self.warmup)
            return 0.5*(1 + math.cos(math.pi * t / T))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.history = []

    def train_one_epoch(self, loader):
        self.model.train()
        run_loss = 0.0
        for batch in loader:
            x = batch["image"].to(self.device)
            y = batch["mask"].to(self.device).long()
            self.optimizer.zero_grad()
            logits = self.model(x)                             # já sai no HxW da entrada
            loss_dict = self.criterion(logits, y)
            loss = loss_dict["total"]
            loss.backward()
            self.optimizer.step()
            run_loss += float(loss.item())
        return run_loss / max(1, len(loader))

    @torch.no_grad()
    def validate(self, loader, tta=True):
        self.model.eval()
        mets = []
        for batch in loader:
            x = batch["image"].to(self.device)
            y = batch["mask"].to(self.device).long()
            if tta:
                outs=[]
                for tf in ["orig","fh","fv","rot90"]:
                    xin = x
                    if tf=="fh":     xin = torch.flip(x, dims=[3])
                    elif tf=="fv":   xin = torch.flip(x, dims=[2])
                    elif tf=="rot90": xin = torch.rot90(x, k=1, dims=[2,3])
                    p = self.model(xin)
                    if tf=="fh":     p = torch.flip(p, dims=[3])
                    elif tf=="fv":   p = torch.flip(p, dims=[2])
                    elif tf=="rot90": p = torch.rot90(p, k=3, dims=[2,3])
                    outs.append(p)
                logits = torch.mean(torch.stack(outs, dim=0), dim=0)
            else:
                logits = self.model(x)
            m = av_pixel_metrics(logits, y)
            mets.append(m)
        macro = float(np.mean([m["macro_f1"] for m in mets]))
        acc   = float(np.mean([m["acc"]      for m in mets]))
        f1a   = float(np.mean([m["f1_art"]   for m in mets]))
        f1v   = float(np.mean([m["f1_vein"]  for m in mets]))
        return {"macro_f1": macro, "acc": acc, "f1_art": f1a, "f1_vein": f1v}

    def fit(self, train_loader, val_loader, run_id: str, ckpt_dir: Path):
        ckpt_dir = Path(ckpt_dir); ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_macro, patience = -1.0, 0
        best_path = ckpt_dir / f"multi_dataset_best_model_{run_id}.pth"

        for ep in range(1, self.epochs+1):
            tloss = self.train_one_epoch(train_loader)
            valm  = self.validate(val_loader, tta=True)
            self.scheduler.step()

            rec = {
                "epoch": ep,
                "train_loss": tloss,
                "val_macro_f1": valm["macro_f1"],
                "val_acc": valm["acc"],
                "val_f1_art": valm["f1_art"],
                "val_f1_vein": valm["f1_vein"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            self.history.append(rec)
            print(f"[AV][ep {ep:03d}] tloss={tloss:.4f} macroF1={valm['macro_f1']:.4f} acc={valm['acc']:.4f} lr={rec['lr']:.2e}")

            if valm["macro_f1"] > best_macro:
                best_macro, patience = valm["macro_f1"], 0
                torch.save({"state_dict": self.model.state_dict()}, best_path)
            else:
                patience += 1
            if patience >= self.early:
                print(f"Early stopping em ep {ep} (patience={self.early})")
                break

        # salvar histórico em CSV
        hist_csv = self.log_dir / f"history_AV_{run_id}.csv"
        with open(hist_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            w.writeheader(); w.writerows(self.history)
        print("Histórico salvo em:", hist_csv, "| Best:", best_path)
        return best_path, hist_csv
