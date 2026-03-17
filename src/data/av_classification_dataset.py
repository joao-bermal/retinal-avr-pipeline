# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, ConcatDataset
from pathlib import Path
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config.settings import AV_CLASSIFICATION_CONFIG

class EnhancedIOSTARDataset(Dataset):
    """Enhanced IOSTAR Dataset - usando processamento original que funciona"""
    
    def __init__(self, base_path, phase='train', transform=None):
        self.base_path = Path(base_path)
        self.phase = phase
        self.transform = transform
        self.img_size = AV_CLASSIFICATION_CONFIG["DATASET"]["IMAGE_SIZE"]
        
        # IOSTAR paths
        self.vessel_dir = self.base_path / 'GT'
        self.av_dir = self.base_path / 'AV_GT'
        
        # Load samples
        self.samples = self.load_samples()
        
    def load_samples(self):
        """Load samples with proper splitting"""
        
        # Find all available pairs
        all_samples = []
        vessel_files = sorted(self.vessel_dir.glob('*.tif'))
        
        for vessel_file in vessel_files:
            base_name = vessel_file.stem.replace('_GT', '')
            
            av_file = self.av_dir / f'{base_name}_AV.tif'
            
            if av_file.exists():
                all_samples.append({
                    'id': base_name,
                    'vessel_path': vessel_file,
                    'av_path': av_file,
                })
        
        return all_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load vessel mask (input)
            vessel_img = cv2.imread(str(sample['vessel_path']), cv2.IMREAD_GRAYSCALE)
            if vessel_img is None:
                raise Exception(f"Error loading vessel: {sample['vessel_path']}")
            
            vessel_rgb = cv2.cvtColor(vessel_img, cv2.COLOR_GRAY2RGB)
            
            # Load AV ground truth
            av_img = cv2.imread(str(sample['av_path']))
            if av_img is None:
                raise Exception(f"Error loading AV: {sample['av_path']}")
            
            # USAR PROCESSAMENTO ORIGINAL QUE FUNCIONA
            av_mask = self.process_av_ground_truth_original(av_img)
            vessel_binary = (vessel_img > 0).astype(np.uint8)
            
            # Resize to target size
            vessel_rgb = cv2.resize(vessel_rgb, (self.img_size, self.img_size))
            av_mask = cv2.resize(av_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            vessel_binary = cv2.resize(vessel_binary, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            
            # Convert to tensors
            vessel_tensor = torch.from_numpy(vessel_rgb.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(av_mask).long()
            
            return {
                'image': vessel_tensor,
                'mask': mask_tensor,
                'dataset': 'iostar'
            }
            
        except Exception as e:
            print(f"Error loading IOSTAR sample {idx}: {e}")
            # Error fallback
            return {
                'image': torch.zeros(3, self.img_size, self.img_size, dtype=torch.float),
                'mask': torch.zeros(self.img_size, self.img_size, dtype=torch.long),
                'dataset': 'iostar'
            }
    
    def process_av_ground_truth_original(self, av_img):
        """Process AV ground truth - CÓDIGO ORIGINAL QUE FUNCIONA"""
        if len(av_img.shape) == 3:
            av_rgb = cv2.cvtColor(av_img, cv2.COLOR_BGR2RGB)
        else:
            av_rgb = cv2.cvtColor(av_img, cv2.COLOR_GRAY2RGB)
        
        h, w = av_rgb.shape[:2]
        classes = np.zeros((h, w), dtype=np.uint8)
        
        # Extract color channels
        r = av_rgb[:, :, 0].astype(np.float32)
        g = av_rgb[:, :, 1].astype(np.float32) 
        b = av_rgb[:, :, 2].astype(np.float32)
        
        tolerance = 20
        
        # Artérias (vermelho)
        artery_mask = (r >= 255 - tolerance) & (g <= tolerance) & (b <= tolerance)
        # Veias (azul)
        vein_mask = (b >= 255 - tolerance) & (r <= tolerance) & (g <= tolerance)
        
        classes[artery_mask] = 1  # Artéria
        classes[vein_mask] = 2    # Veia
        
        return classes

class EnhancedRITEDataset(Dataset):
    """RITE - CORRIGIDO: vessel → av"""
    def __init__(self, base_path, phase='train', transform=None):
        self.base_path = Path(base_path)
        self.phase = phase
        self.transform = transform
        self.img_size = AV_CLASSIFICATION_CONFIG["DATASET"]["IMAGE_SIZE"]
        
        # RITE paths - CORRIGIDO!
        train_vessel_dir = self.base_path / 'training' / 'vessel'  # VESSEL INPUT ***
        train_av_dir = self.base_path / 'training' / 'av'          # A/V TARGET ***
        test_vessel_dir = self.base_path / 'test' / 'vessel'       # VESSEL INPUT ***
        test_av_dir = self.base_path / 'test' / 'av'               # A/V TARGET ***
        
        # Collect all pairs
        self.samples = []
        
        # Training pairs
        for vessel_file in sorted(train_vessel_dir.glob('*.png')):
            base_name = vessel_file.stem  # e.g., "21_training"
            av_file = train_av_dir / f'{base_name}.png'
            if av_file.exists():
                self.samples.append({
                    'vessel_path': str(vessel_file),
                    'av_path': str(av_file),
                    'split': 'train'
                })
        
        # Test pairs  
        for vessel_file in sorted(test_vessel_dir.glob('*.png')):
            base_name = vessel_file.stem  # e.g., "01_test"
            av_file = test_av_dir / f'{base_name}.png'
            if av_file.exists():
                self.samples.append({
                    'vessel_path': str(vessel_file),
                    'av_path': str(av_file), 
                    'split': 'test'
                })
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load vessel mask (input)
            vessel_img = cv2.imread(sample['vessel_path'], cv2.IMREAD_GRAYSCALE)
            if vessel_img is None:
                raise Exception(f"Error loading vessel: {sample['vessel_path']}")
            
            vessel_rgb = cv2.cvtColor(vessel_img, cv2.COLOR_GRAY2RGB)
            
            # Load AV ground truth
            av_img = cv2.imread(sample['av_path'])
            if av_img is None:
                raise Exception(f"Error loading AV: {sample['av_path']}")
            
            av_mask = standardize_av_mask_rite(av_img)
            
            # Resize to target size
            vessel_rgb = cv2.resize(vessel_rgb, (self.img_size, self.img_size))
            av_mask = cv2.resize(av_mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            
            # Convert to tensors
            vessel_tensor = torch.from_numpy(vessel_rgb.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(av_mask).long()
            
            return {
                'image': vessel_tensor,
                'mask': mask_tensor,
                'dataset': 'rite'
            }
            
        except Exception as e:
            print(f"Error loading RITE sample {idx}: {e}")
            # Error fallback
            return {
                'image': torch.zeros(3, self.img_size, self.img_size, dtype=torch.float),
                'mask': torch.zeros(self.img_size, self.img_size, dtype=torch.long),
                'dataset': 'rite'
            }

class EnhancedLESAVDataset(Dataset):
    """LES-AV Dataset - com masks separadas"""
    def __init__(self, base_path, phase='train', transform=None):
        self.base_path = Path(base_path)
        self.phase = phase
        self.transform = transform
        self.img_size = AV_CLASSIFICATION_CONFIG["DATASET"]["IMAGE_SIZE"]
        
        self.images_dir = self.base_path / 'images'
        self.artery_masks_dir = self.base_path / 'artery_masks'
        self.vein_masks_dir = self.base_path / 'vein_masks'
        
        self.samples = self.load_samples()
        
    def load_samples(self):
        all_samples = []
        image_files = sorted(self.images_dir.glob('*.jpg'))
        
        for img_file in image_files:
            base_name = img_file.stem
            artery_mask_file = self.artery_masks_dir / f'{base_name}.png'
            vein_mask_file = self.vein_masks_dir / f'{base_name}.png'
            
            if artery_mask_file.exists() and vein_mask_file.exists():
                all_samples.append({
                    'image_path': str(img_file),
                    'artery_path': str(artery_mask_file),
                    'vein_path': str(vein_mask_file),
                })
        return all_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = cv2.imread(sample['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            av_mask = create_lesav_mask(
                sample['artery_path'], 
                sample['vein_path'], 
                target_size=self.img_size
            )
            
            # Resize image
            image = cv2.resize(image, (self.img_size, self.img_size))
            
            # Apply transformations
            if self.transform:
                augmented = self.transform(image=image, mask=av_mask)
                image = augmented['image']
                av_mask = augmented['mask']
            else:
                image = ToTensorV2()(image=image)['image']
                av_mask = torch.from_numpy(av_mask).long()
            
            return {
                'image': image,
                'mask': av_mask,
                'dataset': 'lesav'
            }
            
        except Exception as e:
            print(f"Error loading LES-AV sample {idx}: {e}")
            return {
                'image': torch.zeros(3, self.img_size, self.img_size, dtype=torch.float),
                'mask': torch.zeros(self.img_size, self.img_size, dtype=torch.long),
                'dataset': 'lesav'
            }

class CombinedAVDataset(ConcatDataset):
    """Combina múltiplos datasets A/V para treinamento multi-dataset."""
    def __init__(self, config, phase='train', transform=None):
        self.config = config
        self.phase = phase
        self.transform = transform
        
        self.datasets = []
        self.dataset_names = []
        self.lengths = []
        self.weights = []
        
        if config['DATASETS']['iostar']['enabled']:
            try:
                iostar_ds = EnhancedIOSTARDataset(
                    config['DATASETS']['iostar']['BASE_PATH'], 
                    phase, 
                    transform
                )
                self.datasets.append(iostar_ds)
                self.dataset_names.append('iostar')
                self.lengths.append(len(iostar_ds))
                self.weights.append(config['DATASETS']['iostar']['weight'])
            except Exception as e:
                print(f"***x*** Failed to load IOSTAR: {e}")
        
        if config['DATASETS']['rite']['enabled']:
            try:
                rite_ds = EnhancedRITEDataset(
                    config['DATASETS']['rite']['BASE_PATH'], 
                    phase, 
                    transform
                )
                self.datasets.append(rite_ds)
                self.dataset_names.append('rite')
                self.lengths.append(len(rite_ds))
                self.weights.append(config['DATASETS']['rite']['weight'])
            except Exception as e:
                print(f"***x*** Failed to load RITE: {e}")
        
        if config['DATASETS']['lesav']['enabled']:
            try:
                lesav_ds = EnhancedLESAVDataset(
                    config['DATASETS']['lesav']['BASE_PATH'], 
                    phase, 
                    transform
                )
                self.datasets.append(lesav_ds)
                self.dataset_names.append('lesav')
                self.lengths.append(len(lesav_ds))
                self.weights.append(config['DATASETS']['lesav']['weight'])
            except Exception as e:
                print(f"***x*** Failed to load LES-AV: {e}")
        
        super().__init__(self.datasets)
        
        # Calculate cumulative lengths for indexing
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = sum(self.lengths)
        
    def get_dataset_distribution(self):
        """
        Get distribution of samples across datasets
        """
        distribution = {}
        for name, length in zip(self.dataset_names, self.lengths):
            distribution[name] = {
                'count': length,
                'percentage': (length / self.total_length) * 100
            }
        return distribution

class EnhancedAVAugmentation:
    """
    Enhanced Augmentation for Multi-Dataset A/V Classification
    Designed to preserve vascular structures while increasing diversity
    """
    def __init__(self, img_size=768, phase='train'):
        self.img_size = img_size
        self.phase = phase
        self.config = AV_CLASSIFICATION_CONFIG["AUGMENTATION"]
        self.transform = self.get_transform()
    
    def get_transform(self):
        if self.phase == 'train':
            return A.Compose([
                # Resize first
                A.Resize(
                    self.img_size, 
                    self.img_size, 
                    interpolation=cv2.INTER_CUBIC
                ),
                
                # Geometric transforms - preserve vessel connectivity
                A.OneOf([
                    A.Rotate(
                        limit=self.config['ROTATION'], 
                        border_mode=cv2.BORDER_REFLECT, 
                        p=0.8
                    ),
                    A.RandomRotate90(p=0.4)
                ], p=0.9),
                
                # Flips
                A.HorizontalFlip(p=self.config['FLIP_PROB']),
                A.VerticalFlip(p=self.config['FLIP_PROB']),
                
                # Color/Intensity transforms
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=self.config['INTENSITY_SHIFT'],
                        contrast_limit=self.config['INTENSITY_SHIFT'],
                        p=0.8
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=10,
                        sat_shift_limit=15,
                        val_shift_limit=10,
                        p=0.5
                    )
                ], p=0.8),
                
                # CLAHE for better vessel contrast
                A.CLAHE(
                    clip_limit=2.0,
                    tile_grid_size=(8, 8),
                    p=self.config['CLAHE_PROB']
                ),
                
                # Blur (light) - simulate focus variations
                A.OneOf([
                    A.GaussianBlur(blur_limit=(1, 3), p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.2),
                ], p=0.2),
                
                # Normalization and tensor conversion
                A.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225), 
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ], p=self.config['PROBABILITY'])
        
        else:  # validation/test
            return A.Compose([
                A.Resize(
                    self.img_size, 
                    self.img_size, 
                    interpolation=cv2.INTER_CUBIC
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225), 
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ])