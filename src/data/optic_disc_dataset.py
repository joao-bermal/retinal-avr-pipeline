import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

# IOSTAR/mask_OD is NOT a simple "disc=255, rest=0" binary mask: value 0
# marks both the optic disc AND the region outside the field of view (FOV),
# which are connected through the image borders (255 = normal retinal
# tissue, most pixels). To isolate just the disc, we discard the connected
# component that touches the border (FOV) and keep the compact
# plausibly-sized component(s) near the image center.
_MIN_DISC_RADIUS_FRACTION = 0.03
_MAX_DISC_RADIUS_FRACTION = 0.25


def _extract_disc_mask(raw_mask: np.ndarray) -> np.ndarray:
    """Isolates the optic disc from the raw IOSTAR mask (see note above)."""
    h, w = raw_mask.shape[:2]
    min_r = _MIN_DISC_RADIUS_FRACTION * min(h, w)
    max_r = _MAX_DISC_RADIUS_FRACTION * min(h, w)

    zero_region = (raw_mask == 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(zero_region, connectivity=8)

    disc_mask = np.zeros_like(zero_region)
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        touches_border = x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h
        radius_est = np.sqrt(area / np.pi)
        if touches_border or not (min_r <= radius_est <= max_r):
            continue
        disc_mask[labels == i] = 1

    if disc_mask.sum() == 0:
        # Fallback: no plausible component isolated -- use the whole ==0
        # region (may include the background outside the FOV, but avoids
        # an empty mask).
        disc_mask = zero_region

    return disc_mask


class OpticDiscDataset(Dataset):
    """
    Optic disc segmentation dataset from IOSTAR (the only downloaded
    dataset with disc mask ground truth -- data/IOSTAR/mask_OD/).

    Pairs data/IOSTAR/image/<id>.jpg with data/IOSTAR/mask_OD/<id>_ODMask.tif.
    Follows the same pattern as EnhancedDRIVEDataset (deterministic 80/20
    split, no shuffling before splitting, for reproducibility).
    """

    def __init__(self, base_path, phase="train", img_size=(512, 512), transform=None):
        self.base_path = Path(base_path)
        self.phase = phase
        self.img_size = img_size
        self.transform = transform

        self.image_dir = self.base_path / "image"
        self.mask_dir = self.base_path / "mask_OD"

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"{self.image_dir} does not exist -- copy the original IOSTAR images "
                "(see EXECUTION_GUIDE.md, 'Original IOSTAR images' section)."
            )

        all_images = sorted(self.image_dir.glob("*.jpg"))
        self.images, self.masks = [], []
        for img_path in all_images:
            mask_path = self.mask_dir / f"{img_path.stem}_ODMask.tif"
            if mask_path.exists():
                self.images.append(img_path)
                self.masks.append(mask_path)
            else:
                print(f"[WARN] optic disc mask not found for {img_path.name}")

        assert len(self.images) == len(self.masks), "Number of images and masks doesn't match."
        assert len(self.images) > 0, f"No image/mask pairs found in {self.base_path}"

        # Deterministic 80/20 split (small dataset: ~30 images).
        split_idx = max(1, int(len(self.images) * 0.8))
        if self.phase == "train":
            self.images = self.images[:split_idx]
            self.masks = self.masks[:split_idx]
        else:
            self.images = self.images[split_idx:] or self.images[-1:]
            self.masks = self.masks[split_idx:] or self.masks[-1:]

        print(f"OpticDiscDataset [{self.phase}]: {len(self.images)} samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        # Isolate the optic disc BEFORE resizing, so the "touches the
        # border" test (which excludes the background outside the FOV)
        # operates at the mask's native resolution.
        disc_mask = _extract_disc_mask(raw_mask)

        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        mask = cv2.resize(
            disc_mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST
        ).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()

        return image, mask
