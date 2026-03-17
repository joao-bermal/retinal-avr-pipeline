from pathlib import Path
import torch, torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.config.settings import SEGMENTATION_CONFIG as C
from src.training.losses import CombinedLossOptimized
from src.metrics.evaluation_metrics import compute_segmentation_metrics

class EnhancedSegmentationTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model.to(C.DEVICE)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.criterion = CombinedLossOptimized()
        self.optimizer = optim.Adam(self.model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=C.LR_PATIENCE, mode="max", factor=0.5)
        self.best_dice, self.patience = -1, 0
        self.best_path = C.MODELS_PATH / "best_model.pth"
        C.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    def train(self, epochs=C.EPOCHS):
        for _ in range(epochs):
            self.model.train()
            for x, y in self.train_loader:
                x, y = x.to(C.DEVICE), y.to(C.DEVICE)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()
            dice, acc, sen, spe, iou = self.validate()
            self.scheduler.step(dice)
            if dice > self.best_dice:
                self.best_dice, self.patience = dice, 0
                torch.save(self.model.state_dict(), self.best_path)
            else:
                self.patience += 1
            if self.patience >= C.EARLY_STOPPING_PATIENCE:
                break
        return self.best_path
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        all_metrics = []
        for x, y in self.val_loader:
            x, y = x.to(C.DEVICE), y.to(C.DEVICE)
            p = torch.sigmoid(self.model(x))
            all_metrics.append(compute_segmentation_metrics(p, y))
        return tuple(sum(m[i] for m in all_metrics)/len(all_metrics) for i in range(5))
