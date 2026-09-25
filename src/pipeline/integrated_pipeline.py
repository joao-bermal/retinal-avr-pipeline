import logging
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path

from src.config.settings import (
    SEGMENTATION_CONFIG as SC,
    AV_CLASSIFICATION_CONFIG as AC,
    OPTIC_DISC_CONFIG as OC,
    PIPELINE_CONFIG as PC,
)
from src.models.segmentation_model import EnhancedUNet
from src.models.av_classification_model import EnhancedMultiDatasetAVNet
from src.data.preprocessing import apply_enhanced_preprocessing
from src.pipeline.avr_calculator import ScientificAVRCalculator
from src.pipeline.optic_disc import OpticDiscDetector, detect_optic_disc

logger = logging.getLogger(__name__)


def _find_latest_checkpoint(models_dir: Path, exclude_dirs=("legacy",)):
    """Resolve o checkpoint .pth mais recente (por mtime) sob `models_dir`."""
    candidates = [
        p for p in Path(models_dir).rglob("*.pth")
        if not any(part in exclude_dirs for part in p.relative_to(models_dir).parts)
    ]
    if not candidates:
        candidates = list(Path(models_dir).rglob("*.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


class ScientificAVRPipeline:
    """
    Pipeline cientifico completo de analise AVR:
    Fundus -> Segmentacao de vasos -> Classificacao A/V -> Deteccao do disco
    optico -> Calculo do AVR (Zona B, Knudtson) -> Interpretacao de risco.
    """

    def __init__(self, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.seg_model = None
        self.av_model = None
        self.od_model = None  # OpticDiscDetector opcional (None = so CV classico)
        self.is_initialized = False

        self.seg_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(SC['DATASET']['IMAGE_SIZE']),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.av_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((AC["DATASET"]["IMAGE_SIZE"], AC["DATASET"]["IMAGE_SIZE"])),
            T.ToTensor(),
        ])

    def load_models(self, seg_path=None, av_path=None, od_path=None):
        """
        Carrega os checkpoints treinados. Ao contrario da versao anterior,
        NAO retorna sucesso silenciosamente se um checkpoint obrigatorio
        (segmentacao/classificacao A/V) nao for encontrado -- isso rodaria
        inferencia com pesos aleatorios sem aviso nenhum.
        """
        seg_ckpt = seg_path or _find_latest_checkpoint(SC['PATHS']['MODELS'])
        if seg_ckpt is None:
            logger.error("Nenhum checkpoint de segmentacao encontrado em %s", SC['PATHS']['MODELS'])
            return False

        self.seg_model = EnhancedUNet(
            in_channels=SC['MODEL']['IN_CHANNELS'],
            out_channels=SC['MODEL']['OUT_CHANNELS'],
            features=SC['MODEL']['FEATURES'],
        ).to(self.device).eval()
        self.seg_model.load_state_dict(torch.load(seg_ckpt, map_location=self.device, weights_only=True))
        logger.info("Modelo de segmentacao carregado de %s", seg_ckpt)

        av_ckpt = av_path or _find_latest_checkpoint(AC['PATHS']['MODELS'])
        if av_ckpt is None:
            logger.error("Nenhum checkpoint de classificacao A/V encontrado em %s", AC['PATHS']['MODELS'])
            return False

        self.av_model = EnhancedMultiDatasetAVNet(AC).to(self.device).eval()
        checkpoint = torch.load(av_ckpt, map_location=self.device, weights_only=False)
        state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        self.av_model.load_state_dict(state)
        logger.info("Modelo de classificacao A/V carregado de %s", av_ckpt)

        # Disco optico e OPCIONAL: se nao houver checkpoint treinado ainda,
        # o pipeline segue com a heuristica de CV classico (ver optic_disc.py).
        od_ckpt = od_path or _find_latest_checkpoint(OC['PATHS']['MODELS'])
        if od_ckpt is not None:
            try:
                self.od_model = OpticDiscDetector(
                    od_ckpt, device=self.device, image_size=OC['DATASET']['IMAGE_SIZE'],
                )
                logger.info("Modelo de disco optico carregado de %s", od_ckpt)
            except Exception:
                logger.exception("Falha ao carregar modelo de disco optico -- usando so CV classico.")
                self.od_model = None
        else:
            logger.info(
                "Nenhum checkpoint de disco optico treinado ainda -- usando heuristica de CV classico "
                "(python main.py --train_od para treinar um modelo)."
            )

        self.is_initialized = True
        return True

    @torch.no_grad()
    def segment_vessels(self, image):
        """Segmenta a arvore vascular. Aplica o mesmo pre-processamento (CLAHE
        + gamma) usado no treino, essencial para a distribuicao de entrada
        bater com a esperada pelo modelo."""
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) if isinstance(image, str) else image
        preprocessed = apply_enhanced_preprocessing(img)

        x = self.seg_transform(preprocessed).unsqueeze(0).to(self.device)
        t0 = time.time()
        mask = self.seg_model(x).squeeze().cpu().numpy()
        ms = (time.time() - t0) * 1000
        bin_mask = (mask > 0.3).astype(np.uint8) * 255
        return {
            "mask": bin_mask,
            "vessel_percentage": float((bin_mask > 0).mean() * 100),
            "inference_time_ms": ms,
        }

    @torch.no_grad()
    def classify_av(self, vessel_mask):
        """Classifica os pixels de vaso em arteria/veia a partir da mascara binaria."""
        if vessel_mask.ndim == 2:
            vessel_mask = np.stack([vessel_mask] * 3, axis=-1)
        x = self.av_transform(vessel_mask).unsqueeze(0).to(self.device)
        t0 = time.time()
        logits = self.av_model(x)
        ms = (time.time() - t0) * 1000
        probabilities = torch.softmax(logits, dim=1)
        preds = torch.argmax(probabilities, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        return {
            "predictions": preds,  # HxW, tamanho AC['DATASET']['IMAGE_SIZE']
            "confidence": float(probabilities.max(dim=1)[0].mean().item()),
            "inference_time_ms": ms,
        }

    def detect_optic_disc(self, image_rgb):
        """Detecta o disco optico na imagem original (coordenadas nativas)."""
        return detect_optic_disc(image_rgb, model=self.od_model, min_confidence=PC.get("OD_MIN_CONFIDENCE", 0.3))

    def calculate_avr(self, artery_mask, vein_mask, image_shape, od_center, od_radius):
        """Calcula o AVR cientifico (Zona B, Knudtson) dado o disco optico ja detectado."""
        calculator = ScientificAVRCalculator(image_shape, optic_disc_center=od_center, optic_disc_radius=od_radius)
        return calculator.calculate_scientific_avr(artery_mask, vein_mask)

    def process_image(self, image_path, verbose=True):
        """Pipeline completo: imagem -> vasos -> A/V -> disco optico -> AVR -> risco."""
        if not self.is_initialized:
            if not self.load_models():
                return {"image_path": str(image_path), "error": "Failed to load models"}

        results = {"image_path": str(image_path)}

        try:
            original_bgr = cv2.imread(str(image_path))
            if original_bgr is None:
                raise ValueError(f"Nao foi possivel ler a imagem: {image_path}")
            original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            h0, w0 = original_rgb.shape[:2]

            # Passo 1: segmentacao de vasos
            seg_results = self.segment_vessels(original_rgb)
            results.update(seg_results)
            if verbose:
                print(f"Segmentacao: {seg_results['inference_time_ms']:.1f}ms, "
                      f"{seg_results['vessel_percentage']:.1f}% de vasos")

            # Passo 2: classificacao A/V (opera sobre a mascara de vasos)
            av_results = self.classify_av(seg_results["mask"])
            results["confidence"] = av_results["confidence"]
            results["av_inference_time_ms"] = av_results["inference_time_ms"]
            if verbose:
                print(f"Classificacao A/V: {av_results['inference_time_ms']:.1f}ms, "
                      f"confianca={av_results['confidence']:.3f}")

            # Passo 3: deteccao do disco optico (na imagem ORIGINAL, nao na mascara)
            od_result = self.detect_optic_disc(original_rgb)
            results["optic_disc_center"] = od_result["center"]
            results["optic_disc_radius"] = od_result["radius"]
            results["optic_disc_method"] = od_result["method"]
            results["optic_disc_confidence"] = od_result["confidence"]
            if verbose:
                print(f"Disco optico: metodo={od_result['method']} "
                      f"centro={od_result['center']} raio={od_result['radius']:.1f}px "
                      f"confianca={od_result['confidence']:.2f}")

            # Passo 4: trazer a predicao A/V (tamanho AV_CLASSIFICATION_CONFIG)
            # de volta para a resolucao ORIGINAL da imagem, para que fique no
            # mesmo espaco de coordenadas do disco optico detectado.
            pred_av = cv2.resize(
                av_results["predictions"], (w0, h0), interpolation=cv2.INTER_NEAREST
            )
            artery_mask = (pred_av == 1).astype(np.uint8) * 255
            vein_mask = (pred_av == 2).astype(np.uint8) * 255

            # Passo 5: AVR cientifico (Zona B, Knudtson) na resolucao original
            avr_results = self.calculate_avr(
                artery_mask, vein_mask, (h0, w0), od_result["center"], od_result["radius"]
            )
            results.update({
                "avr": avr_results["avr"],
                "crae": avr_results["crae"],
                "crve": avr_results["crve"],
                "risk_level": avr_results["risk_level"],
                "risk_description": avr_results.get("risk_description", ""),
                "avr_confidence": avr_results["confidence"],
                "avr_status": avr_results["status"],
                "avr_method": avr_results["method"],
            })
            if verbose:
                print(f"AVR: {avr_results['avr']:.3f} (CRAE={avr_results['crae']:.1f}, "
                      f"CRVE={avr_results['crve']:.1f}) -> {avr_results['risk_level']}")

            return results

        except Exception as e:  # noqa: BLE001 - resultado estruturado em vez de crash
            logger.exception("Erro no pipeline integrado")
            results["error"] = str(e)
            return results
