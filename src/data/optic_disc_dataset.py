import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class OpticDiscDataset(Dataset):
    """
    Dataset de segmentacao do disco optico a partir do IOSTAR (unico dataset
    baixado com mascara de disco -- data/IOSTAR/mask_OD/).

    Pareia data/IOSTAR/image/<id>.jpg com data/IOSTAR/mask_OD/<id>_ODMask.tif.
    Segue o mesmo padrao de EnhancedDRIVEDataset (split determinístico
    80/20, sem embaralhar antes de dividir, para reprodutibilidade).
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
                f"{self.image_dir} nao existe -- copie as imagens originais do IOSTAR "
                "(veja EXECUTION_GUIDE.md, secao 'Disco optico')."
            )

        all_images = sorted(self.image_dir.glob("*.jpg"))
        self.images, self.masks = [], []
        for img_path in all_images:
            mask_path = self.mask_dir / f"{img_path.stem}_ODMask.tif"
            if mask_path.exists():
                self.images.append(img_path)
                self.masks.append(mask_path)
            else:
                print(f"[WARN] mascara de disco optico nao encontrada para {img_path.name}")

        assert len(self.images) == len(self.masks), "Numero de imagens e mascaras nao bate."
        assert len(self.images) > 0, f"Nenhum par imagem/mascara encontrado em {self.base_path}"

        # Split deterministico 80/20 (dataset pequeno: ~30 imagens).
        split_idx = max(1, int(len(self.images) * 0.8))
        if self.phase == "train":
            self.images = self.images[:split_idx]
            self.masks = self.masks[:split_idx]
        else:
            self.images = self.images[split_idx:] or self.images[-1:]
            self.masks = self.masks[split_idx:] or self.masks[-1:]

        print(f"OpticDiscDataset [{self.phase}]: {len(self.images)} amostras")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float()

        return image, mask
