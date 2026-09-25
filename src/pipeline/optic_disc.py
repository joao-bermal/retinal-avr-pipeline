# src/pipeline/optic_disc.py
"""
Deteccao do disco optico (OD), usada para localizar corretamente a Zona B
peripapilar no calculo do AVR (ver src/pipeline/avr_calculator.py).

Estrategia hibrida:
1. `OpticDiscDetector` -- modelo treinado (EnhancedUNet reaproveitado,
   1 canal de saida = probabilidade de disco), quando ha checkpoint
   disponivel em models/optic_disc/.
2. `detect_optic_disc_cv` -- heuristica classica de visao computacional
   (regiao mais clara + maior componente conexo + circulo minimo
   envolvente), sempre disponivel, sem necessidade de treino.
3. `detect_optic_disc` -- despacha entre as duas, com fallback final para o
   centro geometrico da imagem (o mesmo comportamento antigo, mas agora
   sempre logado explicitamente em vez de silencioso).
"""

import logging

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# O disco optico tipicamente ocupa entre ~5% e ~25% da menor dimensao da
# imagem em retinografias de fundo padrao (DRIVE/RITE/IOSTAR). Usado para
# rejeitar deteccoes implausiveis.
MIN_DISC_RADIUS_FRACTION = 0.03
MAX_DISC_RADIUS_FRACTION = 0.25

# Fallback de ultimo recurso (mesma heuristica documentada no TCC, pag. 20).
FALLBACK_RADIUS_FRACTION = 0.15


def _bounds_for_shape(shape):
    h, w = shape[:2]
    min_dim = min(h, w)
    return min_dim * MIN_DISC_RADIUS_FRACTION, min_dim * MAX_DISC_RADIUS_FRACTION


def detect_optic_disc_cv(image_rgb):
    """
    Heuristica classica (sem treino): o disco optico e uma das regioes mais
    claras e mais saturadas em vermelho/amarelo do fundo de olho, e
    aproximadamente circular.

    Args:
        image_rgb: imagem RGB (H, W, 3), uint8.

    Returns:
        dict {center: (x, y), radius: float, confidence: float, method: str}
        ou None se nenhuma regiao plausivel foi encontrada.
    """
    if image_rgb is None or image_rgb.size == 0:
        return None

    h, w = image_rgb.shape[:2]
    min_radius, max_radius = _bounds_for_shape(image_rgb.shape)

    # Canal vermelho: o disco optico e tipicamente a regiao mais brilhante
    # no canal vermelho de uma retinografia colorida (Zhu et al.; ARIA).
    red = image_rgb[:, :, 0].astype(np.float32)

    # Mascara de campo de visao (FOV): ignora fundo preto fora da retina.
    fov_mask = (image_rgb.sum(axis=2) > 15).astype(np.uint8)
    if fov_mask.sum() < 0.05 * h * w:
        fov_mask = np.ones((h, w), dtype=np.uint8)

    red_blurred = cv2.GaussianBlur(red, (0, 0), sigmaX=min_dim_sigma(image_rgb.shape))
    red_blurred = cv2.bitwise_and(
        red_blurred.astype(np.uint8), red_blurred.astype(np.uint8), mask=fov_mask
    )

    # Top ~2% dos pixels mais claros dentro do FOV.
    valid = red_blurred[fov_mask > 0]
    if valid.size == 0:
        return None
    threshold = np.percentile(valid, 98)
    bright_mask = ((red_blurred >= threshold) & (fov_mask > 0)).astype(np.uint8)

    # Limpeza morfologica + maior componente conexo.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
    if num_labels <= 1:
        return None

    # Maior componente (ignorando o rotulo 0 = fundo).
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    component_mask = (labels == largest_label).astype(np.uint8)

    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)

    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    area = cv2.contourArea(contour)
    circle_area = np.pi * radius ** 2
    circularity = area / circle_area if circle_area > 0 else 0.0

    if not (min_radius <= radius <= max_radius):
        logger.debug(
            "OD candidato via CV rejeitado: raio %.1fpx fora do intervalo plausivel [%.1f, %.1f]",
            radius, min_radius, max_radius,
        )
        return None

    confidence = float(np.clip(circularity, 0.0, 1.0))
    return {
        "center": (float(cx), float(cy)),
        "radius": float(radius),
        "confidence": confidence,
        "method": "CV_BRIGHTEST_REGION",
    }


def min_dim_sigma(shape):
    """Sigma do blur gaussiano, proporcional ao tamanho da imagem."""
    return max(3.0, min(shape[:2]) * 0.01)


class OpticDiscDetector:
    """
    Wrapper de inferencia para um EnhancedUNet treinado para segmentar o
    disco optico (1 canal de saida, mesma arquitetura da segmentacao de
    vasos -- ver src/models/segmentation_model.py).
    """

    def __init__(self, checkpoint_path, device=None, image_size=(512, 512), threshold=0.5):
        from src.models.segmentation_model import EnhancedUNet

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.threshold = threshold

        self.model = EnhancedUNet(in_channels=3, out_channels=1).to(self.device).eval()
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def detect(self, image_rgb):
        """
        Args:
            image_rgb: imagem RGB (H, W, 3), uint8, tamanho original.

        Returns:
            dict {center, radius, confidence, method} no espaco de
            coordenadas da imagem original, ou None se o modelo nao
            encontrou nenhuma regiao plausivel.
        """
        h0, w0 = image_rgb.shape[:2]
        min_radius, max_radius = _bounds_for_shape(image_rgb.shape)

        resized = cv2.resize(image_rgb, (self.image_size[1], self.image_size[0]))
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        prob_map = self.model(tensor).squeeze().cpu().numpy()
        mask = (prob_map > self.threshold).astype(np.uint8)

        if mask.sum() == 0:
            return None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)

        (cx, cy), radius = cv2.minEnclosingCircle(contour)

        # Reescala de volta para o tamanho original da imagem.
        scale_x = w0 / self.image_size[1]
        scale_y = h0 / self.image_size[0]
        cx_orig, cy_orig = cx * scale_x, cy * scale_y
        radius_orig = radius * (scale_x + scale_y) / 2.0

        if not (min_radius <= radius_orig <= max_radius):
            logger.debug(
                "OD candidato via modelo rejeitado: raio %.1fpx fora do intervalo plausivel",
                radius_orig,
            )
            return None

        confidence = float(np.clip(prob_map[mask > 0].mean(), 0.0, 1.0))
        return {
            "center": (float(cx_orig), float(cy_orig)),
            "radius": float(radius_orig),
            "confidence": confidence,
            "method": "TRAINED_MODEL",
        }


def detect_optic_disc(image_rgb, model=None, min_confidence=0.3):
    """
    Deteccao hibrida do disco optico: tenta o modelo treinado (se fornecido
    e confiante), cai para a heuristica de CV classica, e so em ultimo caso
    cai no centro geometrico da imagem (fallback documentado, nao clinico).

    Args:
        image_rgb: imagem RGB (H, W, 3), uint8.
        model: instancia opcional de OpticDiscDetector (None = pula direto
            para o CV classico).
        min_confidence: confianca minima para aceitar uma deteccao (do
            modelo ou do CV) antes de cair para o proximo metodo.

    Returns:
        dict {center: (x, y), radius: float, confidence: float, method: str}
    """
    if model is not None:
        try:
            result = model.detect(image_rgb)
            if result is not None and result["confidence"] >= min_confidence:
                return result
            logger.info("Modelo de disco optico com baixa confianca, tentando CV classico.")
        except Exception:
            logger.exception("Falha ao rodar o modelo de disco optico, tentando CV classico.")

    cv_result = detect_optic_disc_cv(image_rgb)
    if cv_result is not None and cv_result["confidence"] >= min_confidence:
        return cv_result

    h, w = image_rgb.shape[:2]
    logger.warning(
        "Nenhum metodo de deteccao do disco optico teve confianca suficiente -- "
        "caindo para o centro geometrico da imagem (fallback nao clinico, "
        "documentado como limitacao no TCC)."
    )
    return {
        "center": (w / 2.0, h / 2.0),
        "radius": min(h, w) * FALLBACK_RADIUS_FRACTION,
        "confidence": 0.0,
        "method": "FALLBACK_IMAGE_CENTER",
    }
