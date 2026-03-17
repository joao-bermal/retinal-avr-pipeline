import time, cv2, numpy as np, torch
import torchvision.transforms as T
from pathlib import Path
from src.config.segmentation_config import EnhancedSegmentationConfig as SC
from src.config.av_classification_config import MULTI_ENHANCED_CONFIG as AC
from src.models.segmentation_model import EnhancedUNet
from src.models.av_classification_model import EnhancedMultiDatasetAVNet

class ScientificAVRPipeline:
    def __init__(self, device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.seg_model = None
        self.av_model  = None
        self.av_accuracy = 0.0
        self.av_f1_score = 0.0
        self.seg_transform = T.Compose([T.ToPILImage(), T.Resize(SC.IMAGE_SIZE), T.ToTensor(),
                                        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        self.av_transform  = T.Compose([T.ToPILImage(), T.Resize((AC["DATASET"]["IMAGE_SIZE"], AC["DATASET"]["IMAGE_SIZE"])), T.ToTensor()])
    def load_models(self, seg_path=None, av_path=None):
        # Segmentation
        self.seg_model = EnhancedUNet().to(self.device).eval()
        seg_ckpt = seg_path or (SC.MODELS_PATH / "best_model.pth")
        if Path(seg_ckpt).exists():
            self.seg_model.load_state_dict(torch.load(seg_ckpt, map_location=self.device))
        # AV Classification
        self.av_model = EnhancedMultiDatasetAVNet(AC).to(self.device).eval()
        av_ckpt = av_path or Path(AC["PATHS"]["MODELS"]) / "multi_dataset_best_model.pth"
        if Path(av_ckpt).exists():
            ckpt = torch.load(av_ckpt, map_location=self.device)
            state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            self.av_model.load_state_dict(state)
            self.av_accuracy = ckpt.get("accuracy", 0.0)
            self.av_f1_score = ckpt.get("f1", 0.0)
        return True
    @torch.no_grad()
    def segment_vessels(self, image):
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB) if isinstance(image, str) else image
        x = self.seg_transform(img).unsqueeze(0).to(self.device)
        t0 = time.time()
        mask = self.seg_model(x).squeeze().cpu().numpy()
        ms = (time.time() - t0)*1000
        bin_mask = (mask > 0.3).astype(np.uint8)*255
        return {"mask": bin_mask, "inference_time_ms": ms}
    @torch.no_grad()
    def classify_av(self, vessel_mask):
        if vessel_mask.ndim == 2:
            vessel_mask = np.stack([vessel_mask]*3, axis=-1)
        x = self.av_transform(vessel_mask).unsqueeze(0).to(self.device)
        t0 = time.time()
        logits = self.av_model(x)
        ms = (time.time() - t0)*1000
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze().cpu().numpy()
        return {"av_mask": preds, "inference_time_ms": ms, "acc_ref": self.av_accuracy, "f1_ref": self.av_f1_score}
