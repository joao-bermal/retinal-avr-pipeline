import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config.settings import SEGMENTATION_CONFIG

class EnhancedDRIVEDataset(Dataset):
    """Dataset para o conjunto DRIVE com aprimoramentos para segmentação de vasos."""
    def __init__(self, base_path, phase="train", transform=None):
        self.base_path = Path(base_path)
        self.phase = phase
        self.transform = transform
        self.img_size = SEGMENTATION_CONFIG["DATASET"]["IMAGE_SIZE"]

        if self.phase == "train":
            self.image_dir = self.base_path / SEGMENTATION_CONFIG["DATASET"]["TRAIN_IMAGES"]
            self.mask_dir = self.base_path / SEGMENTATION_CONFIG["DATASET"]["TRAIN_MASKS"]
        else:
            self.image_dir = self.base_path / SEGMENTATION_CONFIG["DATASET"]["TEST_IMAGES"]
            self.mask_dir = self.base_path / SEGMENTATION_CONFIG["DATASET"]["TEST_MASKS"]
            self.fov_mask_dir = self.base_path / SEGMENTATION_CONFIG["DATASET"]["FOV_MASKS"]

        self.images = sorted(list(self.image_dir.glob("*.tif")))
        self.masks = sorted(list(self.mask_dir.glob("*.tif")))
        if self.phase != "train":
            self.fov_masks = sorted(list(self.fov_mask_dir.glob("*.tif")))

        assert len(self.images) == len(self.masks), "Número de imagens e máscaras não corresponde."
        if self.phase != "train":
            assert len(self.images) == len(self.fov_masks), "Número de imagens e máscaras FOV não corresponde."

        print(f"Enhanced DRIVE {self.phase} dataset: {len(self.images)} samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.phase != "train":
            fov_mask_path = self.fov_masks[idx]
            fov_mask = cv2.imread(str(fov_mask_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.bitwise_and(mask, fov_mask) # Aplicar FOV mask

        # Redimensionar para o tamanho alvo
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)

        # Normalizar máscara para 0 ou 1
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0) # Adicionar dimensão de canal para a máscara

class SegmentationAugmentation:
    """Pipeline de aumento de dados para segmentação de vasos."""
    def __init__(self, img_size=(576, 608), phase="train"):
        self.img_size = img_size
        self.phase = phase
        self.transform = self.get_transform()

    def get_transform(self):
        if self.phase == "train":
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(p=0.2),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:  # val/test
            return A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])