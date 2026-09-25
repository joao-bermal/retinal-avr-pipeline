# src/pipeline/avr_calculator.py
"""
Calculo cientifico do AVR (Arteriolar-to-Venular Ratio) a partir de mascaras
binarias de arteria/veia, seguindo o protocolo Knudtson (Zona B peripapilar,
CRAE/CRVE por equivalencia iterativa).

Portado de retinal-avr-cardiovascular-risk/notebooks/03_integrated_pipeline.ipynb
(classe ScientificAVRCalculator), com duas correcoes em relacao ao original:

1. A combinacao iterativa de CRAE/CRVE agora segue o algoritmo canonico de
   Knudtson et al. (2003) -- combinar a cada passo o MAIOR com o MENOR calibre
   restante, reordenando -- em vez do fold-esquerda ingenuo (sempre index 0 +
   index 1) do notebook original.
2. O centro/raio do disco optico usados para montar a Zona B agora sao
   parametros obrigatorios na pratica: quando nao fornecidos, o fallback para
   o centro geometrico da imagem gera um aviso explicito (era silencioso).
   A deteccao real do disco fica em src/pipeline/optic_disc.py.
"""

import logging

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)

# Protocolo Knudtson: Zona B = anel entre 0.5 DD e 1.0 DD do disco optico.
ZONE_B_INNER_DD = 0.5
ZONE_B_OUTER_DD = 1.0

# Coeficientes de equivalencia de Knudtson et al. (2003).
CRAE_COEFFICIENT = 0.88
CRVE_COEFFICIENT = 0.95

MIN_VESSEL_WIDTH_PX = 2.0
MAX_VESSELS_PER_TYPE = 6
MIN_VESSELS_PER_TYPE = 2


class ScientificAVRCalculator:
    """
    Calcula o AVR a partir de larguras vasculares reais medidas na Zona B
    peripapilar, usando as formulas de equivalencia de Knudtson.
    """

    def __init__(self, image_shape, optic_disc_center=None, optic_disc_radius=None):
        """
        Args:
            image_shape: shape (H, W[, C]) da mascara de arteria/veia.
            optic_disc_center: (x, y) em pixels. Se None, usa o centro
                geometrico da imagem como ultimo recurso (impreciso -- loga
                aviso). Prefira sempre passar o resultado de
                src.pipeline.optic_disc.detect_optic_disc().
            optic_disc_radius: raio do disco em pixels. Se None, estima como
                15% da menor dimensao da imagem (heuristica grosseira, mesma
                ressalva acima).
        """
        self.image_shape = image_shape
        h, w = image_shape[:2]

        if optic_disc_center is None:
            logger.warning(
                "ScientificAVRCalculator sem optic_disc_center real -- "
                "usando o centro geometrico da imagem como fallback. "
                "Isso NAO e clinicamente valido; passe o resultado de "
                "detect_optic_disc()."
            )
            self.od_center = (w // 2, h // 2)
        else:
            self.od_center = optic_disc_center

        if optic_disc_radius is None:
            logger.warning(
                "ScientificAVRCalculator sem optic_disc_radius real -- "
                "estimando como 15%% da menor dimensao da imagem."
            )
            self.od_radius = min(h, w) * 0.15
        else:
            self.od_radius = optic_disc_radius

    def _create_roi_mask(self):
        """Mascara da Zona B: anel entre 0.5 DD e 1.0 DD do disco optico."""
        h, w = self.image_shape[:2]
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - self.od_center[0]) ** 2 + (y - self.od_center[1]) ** 2)

        disc_diameter_px = 2.0 * self.od_radius
        inner_radius = ZONE_B_INNER_DD * disc_diameter_px
        outer_radius = ZONE_B_OUTER_DD * disc_diameter_px

        roi_mask = (distances >= inner_radius) & (distances <= outer_radius)
        return roi_mask.astype(np.uint8)

    @staticmethod
    def _get_vessel_skeleton(vessel_mask, roi_mask):
        """Esqueleto morfologico dos vasos restritos a ROI."""
        vessel_binary = (vessel_mask > 127).astype(np.uint8)
        vessel_in_roi = cv2.bitwise_and(vessel_binary, roi_mask)
        if vessel_in_roi.sum() == 0:
            return np.zeros_like(vessel_in_roi)
        return skeletonize(vessel_in_roi > 0).astype(np.uint8)

    @staticmethod
    def _measure_vessel_widths(vessel_mask, skeleton):
        """Largura = 2x a distancia do ponto do esqueleto ate a borda do vaso."""
        vessel_binary = (vessel_mask > 127).astype(np.uint8)
        if vessel_binary.sum() == 0 or skeleton.sum() == 0:
            return []

        distance_map = distance_transform_edt(vessel_binary)
        ys, xs = np.where(skeleton > 0)
        widths = [2.0 * distance_map[y, x] for y, x in zip(ys, xs)]
        return [w for w in widths if w >= 1.0]

    @staticmethod
    def _combine_knudtson(widths, coefficient):
        """
        Combinacao iterativa canonica de Knudtson: a cada passo, combina o
        MAIOR com o MENOR calibre restante e reinsere o valor combinado,
        ate sobrar um unico calibre equivalente.
        """
        if not widths:
            return 0.0
        if len(widths) == 1:
            return widths[0]

        remaining = sorted(widths)
        while len(remaining) > 1:
            smallest = remaining.pop(0)
            largest = remaining.pop(-1)
            combined = coefficient * np.sqrt(largest ** 2 + smallest ** 2)
            remaining.append(combined)
            remaining.sort()
        return remaining[0]

    def _calculate_crae(self, artery_widths):
        return self._combine_knudtson(artery_widths, CRAE_COEFFICIENT)

    def _calculate_crve(self, vein_widths):
        return self._combine_knudtson(vein_widths, CRVE_COEFFICIENT)

    @staticmethod
    def _interpret_avr(avr):
        """Interpretacao clinica (Wong e Mitchell, 2003; Liew et al., 2023)."""
        if avr >= 0.67:
            return "NORMAL", "Baixo risco cardiovascular", "HIGH"
        elif avr >= 0.60:
            return "BORDERLINE", "Risco moderado - monitoramento recomendado", "MEDIUM"
        else:
            return "HIGH_RISK", "Alto risco cardiovascular - avaliacao clinica necessaria", "HIGH"

    @staticmethod
    def _insufficient_vessels_result(artery_count, vein_count):
        return {
            "avr": 0.0,
            "crae": 0.0,
            "crve": 0.0,
            "status": "INSUFFICIENT_VESSELS",
            "risk_level": "INDETERMINATE",
            "risk_description": (
                f"Vasos insuficientes na Zona B: {artery_count} arterias, "
                f"{vein_count} veias (minimo: {MIN_VESSELS_PER_TYPE} cada)"
            ),
            "confidence": "LOW",
            "measurements": {
                "artery_count": artery_count,
                "vein_count": vein_count,
                "artery_widths": [],
                "vein_widths": [],
            },
            "method": "SCIENTIFIC_KNUDTSON",
        }

    def calculate_scientific_avr(self, artery_mask, vein_mask):
        """
        Calcula o AVR cientifico (protocolo Knudtson) a partir das mascaras
        binarias de arteria e veia (0/255), restrito a Zona B peripapilar.

        Returns:
            dict com avr, crae, crve, status, risk_level, risk_description,
            confidence, measurements e method.
        """
        try:
            roi_mask = self._create_roi_mask()

            artery_skeleton = self._get_vessel_skeleton(artery_mask, roi_mask)
            vein_skeleton = self._get_vessel_skeleton(vein_mask, roi_mask)

            artery_widths = self._measure_vessel_widths(artery_mask, artery_skeleton)
            vein_widths = self._measure_vessel_widths(vein_mask, vein_skeleton)

            artery_widths = [w for w in artery_widths if w >= MIN_VESSEL_WIDTH_PX]
            vein_widths = [w for w in vein_widths if w >= MIN_VESSEL_WIDTH_PX]

            top_arteries = sorted(artery_widths, reverse=True)[:MAX_VESSELS_PER_TYPE]
            top_veins = sorted(vein_widths, reverse=True)[:MAX_VESSELS_PER_TYPE]

            if len(top_arteries) < MIN_VESSELS_PER_TYPE or len(top_veins) < MIN_VESSELS_PER_TYPE:
                return self._insufficient_vessels_result(len(top_arteries), len(top_veins))

            crae = self._calculate_crae(top_arteries)
            crve = self._calculate_crve(top_veins)
            avr = crae / crve if crve > 0 else 0.0

            risk_level, risk_description, confidence = self._interpret_avr(avr)

            return {
                "avr": avr,
                "crae": crae,
                "crve": crve,
                "status": "SUCCESS",
                "risk_level": risk_level,
                "risk_description": risk_description,
                "confidence": confidence,
                "measurements": {
                    "artery_count": len(top_arteries),
                    "vein_count": len(top_veins),
                    "artery_widths": top_arteries,
                    "vein_widths": top_veins,
                    "roi_pixels": int(roi_mask.sum()),
                    "optic_disc_center": self.od_center,
                    "optic_disc_radius": self.od_radius,
                },
                "method": "SCIENTIFIC_KNUDTSON",
            }
        except Exception as e:  # noqa: BLE001 - queremos um resultado estruturado, nao um crash
            logger.exception("Falha no calculo cientifico do AVR")
            return {
                "avr": 0.0,
                "crae": 0.0,
                "crve": 0.0,
                "status": "ERROR",
                "risk_level": "INDETERMINATE",
                "risk_description": f"Erro no calculo: {e}",
                "confidence": "LOW",
                "method": "SCIENTIFIC_KNUDTSON",
            }
