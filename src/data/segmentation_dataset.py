import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
from PIL import Image as PILImage
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config.settings import SEGMENTATION_CONFIG
from src.data.preprocessing import apply_enhanced_preprocessing

class EnhancedDRIVEDataset(Dataset):
    """Dataset for the DRIVE dataset with enhancements for vessel segmentation."""
    def __init__(self, base_path, phase="train", transform=None):
        self.base_path = Path(base_path)
        self.phase = phase
        self.transform = transform
        self.img_size = SEGMENTATION_CONFIG["DATASET"]["IMAGE_SIZE"]

        # Only use training directory for both train and validation since test doesn't have public GT
        self.image_dir = self.base_path / SEGMENTATION_CONFIG["DATASET"]["TRAIN_IMAGES"]
        self.mask_dir = self.base_path / SEGMENTATION_CONFIG["DATASET"]["TRAIN_MASKS"]

        all_images = sorted([f for f in self.image_dir.iterdir() if f.suffix.lower() in ('.tif', '.gif', '.png', '.jpg')])

        # Build mask list by matching image base numbers to mask files
        self.images = []
        self.masks = []
        for img_path in all_images:
            mask_path = self._find_mask(img_path)
            if mask_path is not None:
                self.images.append(img_path)
                self.masks.append(mask_path)
            else:
                print(f"⚠️ Mask not found for {img_path.name}")

        assert len(self.images) == len(self.masks), "Number of images and masks do not match."

        # Pre-determined split to be deterministic (first 80% for train, last 20% for val)
        split_idx = int(len(self.images) * 0.8)

        if self.phase == "train":
            self.images = self.images[:split_idx]
            self.masks = self.masks[:split_idx]
        else:
            self.images = self.images[split_idx:]
            self.masks = self.masks[split_idx:]

        print(f"Enhanced DRIVE {self.phase} dataset: {len(self.images)} samples")

    def _find_mask(self, img_path: Path):
        """Maps image to corresponding mask in DRIVE."""
        base_number = img_path.stem.split('_')[0]
        possible_patterns = [
            f"{base_number}_manual1.gif",
            f"{base_number}_manual.gif",
            f"{base_number}.gif",
            f"{base_number}_manual1.png",
            f"{base_number}_manual1.tif",
        ]
        for pattern in possible_patterns:
            candidate = self.mask_dir / pattern
            if candidate.exists():
                return candidate
        return None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply scientific preprocessing (CLAHE + Gamma)
        image = apply_enhanced_preprocessing(image)

        # Load mask with PIL (robust for .gif)
        mask = np.array(PILImage.open(mask_path))
        if len(mask.shape) == 3:
            mask = np.mean(mask, axis=2)
        mask = mask.astype(np.uint8)

        # Resize to target size
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

        # Binarize mask (>0 per notebook)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask  # No unsqueeze - loss performs squeeze internally

class SegmentationAugmentation:
    """Data augmentation pipeline for vessel segmentation (conservative medical augmentation)."""
    def __init__(self, img_size=(576, 608), phase="train"):
        self.img_size = img_size
        self.phase = phase
        self.transform = self.get_transform()

    def get_transform(self):
        if self.phase == "train":
            return A.Compose([
                # Conservative geometric transformations (notebook)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(
                    limit=90,
                    p=0.3,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                # Subtle photometric adjustments (preserve medical characteristics)
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3,
                ),
                # Standard ImageNet normalization
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:  # val/test
            return A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])