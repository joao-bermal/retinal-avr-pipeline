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
    """Resolves the most recent .pth checkpoint (by mtime) under `models_dir`."""
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
    Full scientific AVR analysis pipeline:
    Fundus -> Vessel segmentation -> A/V classification -> Optic disc
    detection -> AVR calculation (Zone B, Knudtson) -> Risk interpretation.
    """

    def __init__(self, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.seg_model = None
        self.av_model = None
        self.od_model = None  # optional OpticDiscDetector (None = classical CV only)
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
        Loads the trained checkpoints. Unlike the previous version, this
        does NOT silently return success if a required checkpoint
        (segmentation/A-V classification) is missing -- that would run
        inference on randomly-initialized weights with no warning at all.
        """
        seg_ckpt = seg_path or _find_latest_checkpoint(SC['PATHS']['MODELS'])
        if seg_ckpt is None:
            logger.error("No segmentation checkpoint found under %s", SC['PATHS']['MODELS'])
            return False

        self.seg_model = EnhancedUNet(
            in_channels=SC['MODEL']['IN_CHANNELS'],
            out_channels=SC['MODEL']['OUT_CHANNELS'],
            features=SC['MODEL']['FEATURES'],
        ).to(self.device).eval()
        self.seg_model.load_state_dict(torch.load(seg_ckpt, map_location=self.device, weights_only=True))
        logger.info("Segmentation model loaded from %s", seg_ckpt)

        av_ckpt = av_path or _find_latest_checkpoint(AC['PATHS']['MODELS'])
        if av_ckpt is None:
            logger.error("No A/V classification checkpoint found under %s", AC['PATHS']['MODELS'])
            return False

        self.av_model = EnhancedMultiDatasetAVNet(AC).to(self.device).eval()
        checkpoint = torch.load(av_ckpt, map_location=self.device, weights_only=False)
        state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        self.av_model.load_state_dict(state)
        logger.info("A/V classification model loaded from %s", av_ckpt)

        # Optic disc is OPTIONAL: if no trained checkpoint exists yet, the
        # pipeline proceeds with the classical CV heuristic (see optic_disc.py).
        od_ckpt = od_path or _find_latest_checkpoint(OC['PATHS']['MODELS'])
        if od_ckpt is not None:
            try:
                self.od_model = OpticDiscDetector(
                    od_ckpt, device=self.device, image_size=OC['DATASET']['IMAGE_SIZE'],
                )
                logger.info("Optic disc model loaded from %s", od_ckpt)
            except Exception:
                logger.exception("Failed to load optic disc model -- using classical CV only.")
                self.od_model = None
        else:
            logger.info(
                "No trained optic disc checkpoint yet -- using the classical CV heuristic "
                "(run python main.py --train_od to train one)."
            )

        self.is_initialized = True
        return True

    @torch.no_grad()
    def segment_vessels(self, image):
        """Segments the vascular tree. Applies the same preprocessing (CLAHE
        + gamma) used during training, essential for the input distribution
        to match what the model expects."""
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
        """Classifies vessel pixels as artery/vein from the binary mask."""
        if vessel_mask.ndim == 2:
            vessel_mask = np.stack([vessel_mask] * 3, axis=-1)
        x = self.av_transform(vessel_mask).unsqueeze(0).to(self.device)
        t0 = time.time()
        logits = self.av_model(x)
        ms = (time.time() - t0) * 1000
        probabilities = torch.softmax(logits, dim=1)
        preds = torch.argmax(probabilities, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        return {
            "predictions": preds,  # HxW, sized AC['DATASET']['IMAGE_SIZE']
            "confidence": float(probabilities.max(dim=1)[0].mean().item()),
            "inference_time_ms": ms,
        }

    def detect_optic_disc(self, image_rgb):
        """Detects the optic disc in the original image (native coordinates)."""
        return detect_optic_disc(image_rgb, model=self.od_model, min_confidence=PC.get("OD_MIN_CONFIDENCE", 0.3))

    def calculate_avr(self, artery_mask, vein_mask, image_shape, od_center, od_radius):
        """Computes the scientific AVR (Zone B, Knudtson) given an already-detected optic disc."""
        calculator = ScientificAVRCalculator(image_shape, optic_disc_center=od_center, optic_disc_radius=od_radius)
        return calculator.calculate_scientific_avr(artery_mask, vein_mask)

    def process_image(self, image_path, verbose=True):
        """Full pipeline: image -> vessels -> A/V -> optic disc -> AVR -> risk."""
        if not self.is_initialized:
            if not self.load_models():
                return {"image_path": str(image_path), "error": "Failed to load models"}

        results = {"image_path": str(image_path)}

        try:
            original_bgr = cv2.imread(str(image_path))
            if original_bgr is None:
                raise ValueError(f"Could not read image: {image_path}")
            original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
            h0, w0 = original_rgb.shape[:2]

            # Step 1: vessel segmentation
            seg_results = self.segment_vessels(original_rgb)
            results.update(seg_results)
            if verbose:
                print(f"Segmentation: {seg_results['inference_time_ms']:.1f}ms, "
                      f"{seg_results['vessel_percentage']:.1f}% vessels")

            # Step 2: A/V classification (operates on the vessel mask)
            av_results = self.classify_av(seg_results["mask"])
            results["confidence"] = av_results["confidence"]
            results["av_inference_time_ms"] = av_results["inference_time_ms"]
            if verbose:
                print(f"A/V classification: {av_results['inference_time_ms']:.1f}ms, "
                      f"confidence={av_results['confidence']:.3f}")

            # Step 3: optic disc detection (on the ORIGINAL image, not the mask)
            od_result = self.detect_optic_disc(original_rgb)
            results["optic_disc_center"] = od_result["center"]
            results["optic_disc_radius"] = od_result["radius"]
            results["optic_disc_method"] = od_result["method"]
            results["optic_disc_confidence"] = od_result["confidence"]
            if verbose:
                print(f"Optic disc: method={od_result['method']} "
                      f"center={od_result['center']} radius={od_result['radius']:.1f}px "
                      f"confidence={od_result['confidence']:.2f}")

            # Step 4: bring the A/V prediction (sized per AV_CLASSIFICATION_CONFIG)
            # back to the ORIGINAL image resolution, so it shares the same
            # coordinate space as the detected optic disc.
            pred_av = cv2.resize(
                av_results["predictions"], (w0, h0), interpolation=cv2.INTER_NEAREST
            )
            artery_mask = (pred_av == 1).astype(np.uint8) * 255
            vein_mask = (pred_av == 2).astype(np.uint8) * 255

            # Step 5: scientific AVR (Zone B, Knudtson) at the original resolution
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

        except Exception as e:  # noqa: BLE001 - structured result instead of a crash
            logger.exception("Error in the integrated pipeline")
            results["error"] = str(e)
            return results
