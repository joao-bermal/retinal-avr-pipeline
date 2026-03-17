import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.config.settings import AV_CLASSIFICATION_CONFIG

class VesselConstraintModule(nn.Module):
    """Vessel-Constraint Module - forces attention on vessel regions"""
    def __init__(self, channels):
        super(VesselConstraintModule, self).__init__()
        self.conv_refine = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, padding=1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//4, channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, features, vessel_mask=None):
        if vessel_mask is not None:
            # Resize vessel mask to feature size
            if vessel_mask.shape[-2:] != features.shape[-2:]:
                vessel_mask = F.interpolate(
                    vessel_mask.unsqueeze(1).float(),
                    size=features.shape[-2:],
                    mode='nearest'
                ).squeeze(1)
            
            # Generate attention weights
            attention_weights = self.conv_refine(features)
            
            # Apply vessel constraint
            vessel_constraint = vessel_mask.unsqueeze(1).expand_as(attention_weights)
            constrained_attention = attention_weights * vessel_constraint
            
            # Apply to features
            enhanced_features = features * (1 + constrained_attention)
            return enhanced_features
        else:
            return features

class Res2NetBlock(nn.Module):
    """Res2Net block for multi-scale receptive fields"""
    def __init__(self, in_channels, out_channels, stride=1, scale=4):
        super(Res2NetBlock, self).__init__()
        self.scale = scale
        width = out_channels // scale
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False) 
            for _ in range(scale-1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(scale-1)])
        
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Split into scale groups
        spx = torch.split(out, out.size(1) // self.scale, 1)
        
        for i in range(self.scale - 1):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        
        out = torch.cat((out, spx[self.scale-1]), 1)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        out = self.relu(out)
        
        return out

class SqueezeExciteBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EnhancedDecoderBlock(nn.Module):
    """Enhanced decoder with all features"""
    def __init__(self, in_channels, skip_channels, out_channels, use_res2net=True):
        super(EnhancedDecoderBlock, self).__init__()
        
        # Upsampling
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=4, stride=2, padding=1, bias=False
        )
        
        # Feature fusion
        fusion_channels = out_channels + skip_channels
        
        if use_res2net:
            self.fusion = Res2NetBlock(fusion_channels, out_channels)
        else:
            self.fusion = nn.Sequential(
                nn.Conv2d(fusion_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # Attention
        self.attention = SqueezeExciteBlock(out_channels)
        
        # VC Module
        self.vc_module = VesselConstraintModule(out_channels)
        
        self.dropout = nn.Dropout2d(AV_CLASSIFICATION_CONFIG["MODEL"]["DROPOUT"])
    
    def forward(self, x, skip, vessel_mask=None):
        # Upsample
        x = self.upsample(x)
        
        # Ensure same size
        if x.size() != skip.size():
            x = F.interpolate(x, size=skip.size()[2:], mode='bilinear', align_corners=False)
        
        # Concatenate and fuse
        x = torch.cat([x, skip], dim=1)
        x = self.fusion(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Apply vessel constraint if available
        x = self.vc_module(x, vessel_mask)
        
        x = self.dropout(x)
        return x

class EnhancedMultiDatasetAVNet(nn.Module):
    """Enhanced AV Network for Multi-Dataset Training"""
    
    def __init__(self, config=AV_CLASSIFICATION_CONFIG):
        super(EnhancedMultiDatasetAVNet, self).__init__()
        
        self.config = config
        self.num_classes = config["DATASET"]["CLASSES"]
        
        # ENCODER - ResNet-50 pretrained
        self.backbone = models.resnet50(pretrained=config["MODEL"]["PRETRAINED"])
        
        # Extract encoder layers
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
        
        # DECODER with enhanced features
        decoder_features = config["MODEL"]["DECODER_FEATURES"]
        
        self.decoder4 = EnhancedDecoderBlock(2048, 1024, decoder_features[0])
        self.decoder3 = EnhancedDecoderBlock(decoder_features[0], 512, decoder_features[1])
        self.decoder2 = EnhancedDecoderBlock(decoder_features[1], 256, decoder_features[2])
        self.decoder1 = EnhancedDecoderBlock(decoder_features[2], 64, decoder_features[3])
        
        # Final classification head
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_features[3], decoder_features[3]//2, 3, padding=1),
            nn.BatchNorm2d(decoder_features[3]//2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(config["MODEL"]["DROPOUT"]),
            nn.Conv2d(decoder_features[3]//2, self.num_classes, 1)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Enhanced weight initialization"""
        for module in self.modules():
            if isinstance(module, (nn.ConvTranspose2d, nn.Conv2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x, vessel_mask=None):
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got {x.dim()}D")
        
        original_size = x.size()[2:]
        
        # ENCODER PATH
        x0 = self.relu(self.bn1(self.conv1(x)))
        x0_pool = self.maxpool(x0)
        
        x1 = self.layer1(x0_pool)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # DECODER PATH with vessel constraint
        d4 = self.decoder4(x4, x3, vessel_mask)
        d3 = self.decoder3(d4, x2, vessel_mask)
        d2 = self.decoder2(d3, x1, vessel_mask)
        d1 = self.decoder1(d2, x0, vessel_mask)
        
        # Final upsampling to original resolution
        d1 = F.interpolate(d1, size=original_size, mode='bilinear', align_corners=False)
        
        # Final classification
        output = self.final_conv(d1)
        
        return output
    
    def create_vessel_mask_from_prediction(self, av_prediction):
        """Create vessel mask from A/V prediction for VC-Module"""
        # av_prediction shape: B, num_classes, H, W
        vessel_mask = torch.sum(av_prediction[:, 1:], dim=1)  # Sum artery + vein channels
        vessel_mask = torch.clamp(vessel_mask, 0, 1)  # Ensure [0,1] range
        return vessel_mask