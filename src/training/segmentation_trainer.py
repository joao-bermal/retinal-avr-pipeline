from pathlib import Path
import time
import torch, torch.optim as optim
import logging
import csv
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.config.settings import SEGMENTATION_CONFIG, DEVICE
from src.training.losses import CombinedLossOptimized
from src.metrics.evaluation_metrics import compute_segmentation_metrics

class EnhancedSegmentationTrainer:
    def __init__(self, model, train_loader, val_loader, resume=False, config=None,
                 keep_all_checkpoints=False, batch_pause_seconds=0.0):
        # `config` lets this generic trainer be reused for other binary
        # segmentation tasks (e.g. optic disc, see OPTIC_DISC_CONFIG) without
        # duplicating the training loop.
        # `keep_all_checkpoints=True` keeps the .pth of every new best epoch
        # (instead of deleting the previous one), used when you want to
        # evaluate multiple epochs against a metric beyond validation Dice
        # (e.g. optic disc center-to-center distance).
        # `batch_pause_seconds` inserts a pause after every training batch:
        # small/fast datasets (e.g. optic disc, 24 images) feed the GPU with
        # no natural data-loading gap between batches, sustaining continuous
        # 100% utilization; on hardware with marginal thermal dissipation
        # this can trigger temperature spikes that don't appear on larger
        # datasets (more I/O/augmentation time per batch creates natural
        # pauses). Cost: slower training, with no effect on the final result.
        C = config or SEGMENTATION_CONFIG
        self.config = C
        self.keep_all_checkpoints = keep_all_checkpoints
        self.batch_pause_seconds = batch_pause_seconds
        self.model = model.to(DEVICE)
        self.train_loader, self.val_loader = train_loader, val_loader
        self.criterion = CombinedLossOptimized()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=C['TRAINING']['LEARNING_RATE'],
            weight_decay=C['TRAINING']['WEIGHT_DECAY'],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            patience=C['TRAINING']['SCHEDULER_PATIENCE'],
            mode="max",
            factor=C['TRAINING']['SCHEDULER_FACTOR'],
            min_lr=1e-7,
        )
        self.best_dice, self.patience = -1, 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}"
        
        # Directories
        self.log_dir = C['PATHS']['LOGS'] / self.run_id
        self.model_dir = C['PATHS']['MODELS'] / self.run_id
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_path = None # Will be set on first improvement

        # Load weights logic here might be overridden if resume is used, but for now we look for the best model in the base path if we resume
        if resume:
            resume_path = C['PATHS']['MODELS'] / "best_model.pth"
            if resume_path.exists():
                print(f"🔄 Loading weights from: {resume_path}")
                self.model.load_state_dict(torch.load(resume_path, map_location=DEVICE, weights_only=True))

        # ===================== LOGGING SETUP =====================
        # Text logger (human readable)
        self.logger = logging.getLogger(f"segmentation_{timestamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        fh = logging.FileHandler(self.log_dir / f"training_{timestamp}.log")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(fh)

        # CSV logger (for analysis/plots)
        self.csv_path = self.log_dir / f"metrics_{timestamp}.csv"
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_acc", "val_sensitivity", "val_specificity", "val_iou", "lr", "best_dice"])

        # Log initial config
        self.logger.info("=" * 60)
        self.logger.info("SEGMENTATION TRAINING STARTED")
        self.logger.info("=" * 60)
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Device: {DEVICE}")
        if DEVICE.type == 'cuda':
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        self.logger.info(f"Model: {C['MODEL']['NAME']}")
        self.logger.info(f"Image size: {C['DATASET']['IMAGE_SIZE']}")
        self.logger.info(f"Batch size: {C['TRAINING']['BATCH_SIZE']}")
        self.logger.info(f"Learning rate: {C['TRAINING']['LEARNING_RATE']}")
        self.logger.info(f"Epochs: {C['TRAINING']['EPOCHS']}")
        self.logger.info(f"Loss weights: Dice={C['LOSS']['DICE_WEIGHT']}, Focal={C['LOSS']['FOCAL_WEIGHT']}, BCE={C['LOSS']['BCE_WEIGHT']}, IoU={C['LOSS']['IOU_WEIGHT']}")
        self.logger.info("-" * 60)
        
        print(f"  📁 Run logs: {self.log_dir}")
        print(f"  📁 Run models: {self.model_dir}")

    def train(self, epochs=None):
        if epochs is None:
            epochs = self.config['TRAINING']['EPOCHS']
            
        history = {
            'train_loss': [], 'val_loss': [], 
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': [],
            'learning_rates': []
        }
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            self.optimizer.zero_grad()
            for i, (x, y) in enumerate(self.train_loader):
                if i == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Batch {i+1}/{len(self.train_loader)}")
                x, y = x.to(DEVICE), y.to(DEVICE)

                logits = self.model(x)

                loss_out = self.criterion(logits, y)
                if isinstance(loss_out, tuple):
                    loss = loss_out[0]
                else:
                    loss = loss_out

                loss.backward()

                # Gradient clipping for stability (notebook: max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config['TRAINING']['GRADIENT_CLIPPING'],
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                train_loss += loss.item()

                if self.batch_pause_seconds > 0:
                    time.sleep(self.batch_pause_seconds)

            train_loss /= len(self.train_loader)

            val_loss, dice, acc, sen, spe, iou = self.validate()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_dice'].append(0.0) # we don't compute train_dice during train to save time, but to keep plots happy we fake it or compute it if needed
            history['val_dice'].append(dice)
            history['train_iou'].append(0.0)
            history['val_iou'].append(iou)
            history['learning_rates'].append(current_lr)

            # Console output
            msg = f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - val_dice: {dice:.4f} - val_iou: {iou:.4f} - LR: {current_lr:.2e}"
            print(msg)

            # File log
            self.logger.info(f"Epoch {epoch+1}/{epochs} | t_loss={train_loss:.4f} | v_loss={val_loss:.4f} | dice={dice:.4f} | acc={acc:.4f} | sen={sen:.4f} | spe={spe:.4f} | iou={iou:.4f} | lr={current_lr:.2e}")

            # CSV log
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{dice:.6f}", f"{acc:.6f}", f"{sen:.6f}", f"{spe:.6f}", f"{iou:.6f}", f"{current_lr:.2e}", f"{self.best_dice:.6f}"])

            self.scheduler.step(dice)
            if dice > self.best_dice:
                self.best_dice, self.patience = dice, 0
                
                # Delete previous best model in this run if it exists
                # (unless keep_all_checkpoints asks to keep the full history)
                if not self.keep_all_checkpoints and self.best_path and self.best_path.exists():
                    self.best_path.unlink()

                self.best_path = self.model_dir / f"model_epoch{epoch+1:03d}_dice_{dice:.4f}.pth"
                torch.save(self.model.state_dict(), self.best_path)
                
                self.logger.info(f"  🏆 NEW BEST MODEL! Dice: {dice:.4f}")
                print(f"  🏆 New best model saved! {self.best_path.name}")
            else:
                self.patience += 1
            if self.patience >= self.config['TRAINING']['EARLY_STOPPING_PATIENCE']:
                self.logger.info(f"  ⏹️ Early stopping at epoch {epoch+1} (patience={self.config['TRAINING']['EARLY_STOPPING_PATIENCE']})")
                print(f"  ⏹️ Early stopping after {self.config['TRAINING']['EARLY_STOPPING_PATIENCE']} epochs without improvement")
                break

        self.logger.info("=" * 60)
        self.logger.info(f"TRAINING COMPLETE | Best Dice: {self.best_dice:.4f}")
        self.logger.info("=" * 60)
        
        # Save results dict
        final_metrics = {'dice_score': self.best_dice} # can load specific if needed
        return self.best_path, history, final_metrics, self.run_id

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        all_metrics = []
        total_loss = 0.0
        for x, y in self.val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Model already applies sigmoid
            p = self.model(x)
            
            # Loss computation
            loss_out = self.criterion(p, y)
            if isinstance(loss_out, tuple):
                loss = loss_out[0]
            else:
                loss = loss_out
            total_loss += loss.item()
            
            # Squeeze channel dim if present
            if p.dim() == 4 and p.size(1) == 1:
                p = p.squeeze(1)
            all_metrics.append(compute_segmentation_metrics(p, y))
        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = tuple(sum(m[i] for m in all_metrics)/len(all_metrics) for i in range(5))
        return (avg_loss, *avg_metrics)
