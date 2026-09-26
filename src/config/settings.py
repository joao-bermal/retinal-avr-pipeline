import torch
from pathlib import Path

# =============================================================================
# GLOBAL CONFIGURATIONS
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# VESSEL SEGMENTATION CONFIGURATIONS (Enhanced U-Net)
# =============================================================================

SEGMENTATION_CONFIG = {
    'DATASET': {
        'NAME': 'DRIVE',
        'IMAGE_SIZE': (576, 608), # Height, Width
        'NUM_CLASSES': 1,
        'BASE_PATH': 'data/DRIVE',
        'TRAIN_IMAGES': 'training/images',
        'TRAIN_MASKS': 'training/1st_manual',
        'TEST_IMAGES': 'test/images',
        'TEST_MASKS': 'test/1st_manual',
        'FOV_MASKS': 'test/mask',
    },
    'MODEL': {
        'NAME': 'EnhancedUNet',
        'IN_CHANNELS': 3,
        'OUT_CHANNELS': 1,
        'FEATURES': [64, 128, 256, 512],
    },
    'TRAINING': {
        'EPOCHS': 150,
        'BATCH_SIZE': 4,
        'LEARNING_RATE': 5e-4,
        'WEIGHT_DECAY': 1e-4,
        'EARLY_STOPPING_PATIENCE': 25,
        'GRADIENT_CLIPPING': 1.0,
        'OPTIMIZER': 'Adam',
        'SCHEDULER': 'ReduceLROnPlateau',
        'SCHEDULER_FACTOR': 0.7,
        'SCHEDULER_PATIENCE': 12,
    },
    'LOSS': {
        'NAME': 'CombinedLoss',
        'DICE_WEIGHT': 0.4,
        'FOCAL_WEIGHT': 0.3,
        'BCE_WEIGHT': 0.2,
        'IOU_WEIGHT': 0.1,
    },
    'TARGETS': {
        'DICE_SCORE': 0.7965,
        'ACCURACY': 0.96,
        'SENSITIVITY': 0.75,
        'SPECIFICITY': 0.98,
        'IOU': 0.6618,
    },
    'PATHS': {
        'MODELS': Path('models/segmentation'),
        'RESULTS': Path('results/segmentation'),
        'LOGS': Path('logs/segmentation'),
        'EVIDENCE': Path('results/segmentation/evidence'),
    },
    'VISUALIZATION': {
        'SAVE_FREQUENCY': 10,
    }
}

# Create directories if they do not exist
for key in SEGMENTATION_CONFIG['PATHS']:
    SEGMENTATION_CONFIG['PATHS'][key].mkdir(parents=True, exist_ok=True)

# =============================================================================
# A/V CLASSIFICATION CONFIGURATIONS (Enhanced Multi-Dataset AV-Net)
# =============================================================================

AV_CLASSIFICATION_CONFIG = {
    'DATASET': {
        'NAME': 'MultiDatasetAV',
        'IMAGE_SIZE': 768,
        'CLASSES': 3,  # 0: Background, 1: Artery, 2: Vein
        'DATASETS': {
            'iostar': {
                'enabled': True,
                'BASE_PATH': 'data/IOSTAR',
                'weight': 1.0
            },
            'rite': {
                'enabled': True,
                'BASE_PATH': 'data/RITE',
                'weight': 1.0
            },
            'lesav': {
                'enabled': True,
                'BASE_PATH': 'data/LES-AV',
                'weight': 1.0
            }
        }
    },
    'AUGMENTATION': {
        'PROBABILITY': 0.8,
        'ROTATION': 30,
        'FLIP_PROB': 0.5,
        'INTENSITY_SHIFT': 0.2,
        'CLAHE_PROB': 0.5
    },
    'MODEL': {
        'NAME': 'MultiDatasetEnhancedAVNet',
        'BACKBONE': 'resnet50',
        'PRETRAINED': True,
        'DROPOUT': 0.1,
        'DECODER_FEATURES': [256, 128, 64, 32],
        'VC_MODULE': True,
        'RES2NET': True,
        'SE_ATTENTION': True
    },
    'TRAINING': {
        'EPOCHS': 250,
        'BATCH_SIZE': 2,
        'LEARNING_RATE': 5e-4,
        'WEIGHT_DECAY': 1e-4,
        'GRADIENT_CLIPPING': 0.5,
        'EARLY_STOPPING_PATIENCE': 30,
        'WARMUP_EPOCHS': 5,
        'BALANCED_SAMPLING': True
    },
    'LOSS': {
        'NAME': 'EnhancedMultiLoss',
        'CLASS_WEIGHTS': [1.0, 5.0, 5.0], # Background, Artery, Vein
        'BCE_WEIGHT': 0.4,
        'DICE_WEIGHT': 0.4,
        'FOCAL_WEIGHT': 0.2,
        'C3_WEIGHT': 0.1,
        'INTRA_WEIGHT': 0.1,
    },
    'TARGETS': {
        'ACCURACY': 0.96,
        'MACRO_F1': 0.78,
        'ARTERY_F1': 0.75,
        'VEIN_F1': 0.80,
    },
    'PATHS': {
        'MODELS': Path('models/av_classification'),
        'RESULTS': Path('results/av_classification'),
        'LOGS': Path('logs/av_classification'),
    }
}

# Create directories if they do not exist
for key in AV_CLASSIFICATION_CONFIG['PATHS']:
    AV_CLASSIFICATION_CONFIG['PATHS'][key].mkdir(parents=True, exist_ok=True)

# =============================================================================
# OPTIC DISC DETECTION CONFIGURATIONS (Enhanced U-Net, 1 channel)
# =============================================================================
# Reuses the EnhancedUNet architecture (same as vessel segmentation), trained
# on data/IOSTAR/mask_OD (the only downloaded dataset with optic disc mask
# ground truth). Used to correctly build the peripapillary Zone B in the AVR
# calculation (see src/pipeline/optic_disc.py and src/pipeline/avr_calculator.py).

OPTIC_DISC_CONFIG = {
    'DATASET': {
        'NAME': 'IOSTAR_OpticDisc',
        'IMAGE_SIZE': (512, 512),  # Height, Width
        'NUM_CLASSES': 1,
        'BASE_PATH': 'data/IOSTAR',
    },
    'MODEL': {
        'NAME': 'EnhancedUNet',
        'IN_CHANNELS': 3,
        'OUT_CHANNELS': 1,
        'FEATURES': [64, 128, 256, 512],
    },
    'TRAINING': {
        # Small dataset (~30 images) -- few epochs, aggressive early stopping.
        'EPOCHS': 80,
        'BATCH_SIZE': 2,
        'LEARNING_RATE': 5e-4,
        'WEIGHT_DECAY': 1e-4,
        'EARLY_STOPPING_PATIENCE': 15,
        'GRADIENT_CLIPPING': 1.0,
        'OPTIMIZER': 'Adam',
        'SCHEDULER': 'ReduceLROnPlateau',
        'SCHEDULER_FACTOR': 0.7,
        'SCHEDULER_PATIENCE': 6,
    },
    'LOSS': {
        'NAME': 'CombinedLoss',
        'DICE_WEIGHT': 0.4,
        'FOCAL_WEIGHT': 0.3,
        'BCE_WEIGHT': 0.2,
        'IOU_WEIGHT': 0.1,
    },
    'TARGETS': {
        'DICE_SCORE': 0.85,  # the optic disc is a simple/compact shape -> higher target than thin vessels
    },
    'PATHS': {
        'MODELS': Path('models/optic_disc'),
        'RESULTS': Path('results/optic_disc'),
        'LOGS': Path('logs/optic_disc'),
        'EVIDENCE': Path('results/optic_disc/evidence'),
    },
    'VISUALIZATION': {
        'SAVE_FREQUENCY': 10,
    }
}

for key in OPTIC_DISC_CONFIG['PATHS']:
    OPTIC_DISC_CONFIG['PATHS'][key].mkdir(parents=True, exist_ok=True)

# =============================================================================
# INTEGRATED PIPELINE CONFIGURATIONS (AVR)
# =============================================================================

PIPELINE_CONFIG = {
    'SEGMENTATION_MODEL_PATH': SEGMENTATION_CONFIG['PATHS']['MODELS'] / 'best_model.pth',
    'AV_CLASSIFICATION_MODEL_PATH': AV_CLASSIFICATION_CONFIG['PATHS']['MODELS'] / 'best_model.pth',
    'OPTIC_DISC_MODEL_PATH': OPTIC_DISC_CONFIG['PATHS']['MODELS'] / 'best_model.pth',
    'AVR_MIN_VESSEL_THRESHOLD': 0.1, # Minimum percentage of vessels to calculate AVR
    'OD_DETECTION_METHOD': 'hybrid',  # 'hybrid' | 'trained_model' | 'cv' | 'fallback_center'
    'OD_MIN_CONFIDENCE': 0.3,
    'ZONE_B_INNER_DD': 0.5,  # Knudtson protocol: annulus between 0.5 and 1.0 disc diameters
    'ZONE_B_OUTER_DD': 1.0,
    'OUTPUT_DIR': Path('results/integrated_pipeline'),
}

PIPELINE_CONFIG['OUTPUT_DIR'].mkdir(parents=True, exist_ok=True)

print(f"Loaded configurations. Device: {DEVICE}")
