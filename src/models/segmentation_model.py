import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Bloco residual para Enhanced U-Net"""
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Inicialização Xavier
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class EnhancedConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_residual: bool = False):
        super(EnhancedConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.use_residual = use_residual and out_channels >= 128
        if self.use_residual:
            self.residual = ResidualBlock(out_channels)
        
        # Inicialização
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
        
    def forward(self, x):
        x = self.conv(x)
        if self.use_residual:
            x = self.residual(x)
        return x

class EnhancedUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features=None):
        super(EnhancedUNet, self).__init__()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        self.features = features
        
        # ENCODER
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        in_ch = in_channels
        for feature in features:
            self.encoder.append(
                EnhancedConvBlock(in_ch, feature, use_residual=True)
            )
            in_ch = feature
        
        # BOTTLENECK
        self.bottleneck = EnhancedConvBlock(
            features[-1], features[-1] * 2, use_residual=True
        )
        self.dropout = nn.Dropout2d(0.5)
        
        # DECODER
        self.upsamples = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        for feature in reversed(features):
            self.upsamples.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(
                EnhancedConvBlock(feature * 2, feature, use_residual=True)
            )
        
        # CLASSIFICAÇÃO FINAL
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        nn.init.xavier_uniform_(self.final_conv.weight)
        
    def forward(self, x):
        # ENCODER PASS
        skip_connections = []
        
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # BOTTLENECK
        x = self.bottleneck(x)
        x = self.dropout(x)
        
        # DECODER PASS
        skip_connections = skip_connections[::-1]
        
        for idx in range(len(self.decoder)):
            x = self.upsamples[idx](x)
            skip_connection = skip_connections[idx]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(
                    x, size=skip_connection.shape[2:],
                    mode='bilinear', align_corners=False
                )
            
            concat_skip = torch.cat([skip_connection, x], dim=1)
            x = self.decoder[idx](concat_skip)
        
        x = self.final_conv(x)
        x = self.sigmoid(x)
        
        return x
