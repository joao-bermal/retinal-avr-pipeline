import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

# O IOSTAR/mask_OD NAO e uma mascara binaria "disco=255, resto=0": o valor 0
# marca tanto o disco optico quanto a regiao fora do campo de visao (FOV),
# que ficam conectados pelas bordas da imagem (255 = tecido retiniano normal,
# a maioria dos pixels). Para isolar so o disco, descartamos o componente
# conexo que toca a borda (FOV) e ficamos com o(s) componente(s) compacto(s)
# de tamanho plausivel no centro da imagem.
_MIN_DISC_RADIUS_FRACTION = 0.03
_MAX_DISC_RADIUS_FRACTION = 0.25


def _extract_disc_mask(raw_mask: np.ndarray) -> np.ndarray:
    """Isola o disco optico a partir da mascara bruta do IOSTAR (ver nota acima)."""
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
        # Fallback: nenhum componente plausivel isolado -- usa a regiao ==0
        # inteira (pode incluir o fundo fora do FOV, mas evita mascara vazia).
        disc_mask = zero_region

    return disc_mask


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

        raw_mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        # Isola o disco optico ANTES de redimensionar, para que o teste de
        # "toca a borda" (que exclui o fundo fora do FOV) opere na resolucao
        # nativa da mascara.
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
