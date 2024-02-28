import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=6, embed_dims=256, num_layers=3, num_classes=5, dropout_prob=.5):
        super(UNet, self).__init__()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Create down blocks
        current_channels = in_channels
        for i in range(num_layers):
            self.down_blocks.append(ConvBlock(current_channels, embed_dims * (2 ** i), dropout_prob))
            current_channels = embed_dims * (2 ** i)
        
        # Bottleneck
        self.bottleneck = ConvBlock(current_channels, current_channels * 2, dropout_prob)
        current_channels *= 2
        
        # Create up blocks
        for i in range(num_layers-1, -1, -1):
            self.up_blocks.append(nn.ConvTranspose1d(current_channels, current_channels // 2, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(current_channels, embed_dims * (2 ** i), dropout_prob))
            current_channels = embed_dims * (2 ** i)
        
        # Final convolution
        self.final_conv = nn.Conv1d(current_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        enc_features = []
        for block in self.down_blocks:
            x = block(x)
            enc_features.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        
        for i in range(0, len(self.up_blocks), 2):
            x = self.up_blocks[i](x)
            enc_feat = enc_features.pop()  # Get the corresponding feature from the encoder
            # Calculate padding
            diffY = enc_feat.size()[2] - x.size()[2]
            x = F.pad(x, [diffY // 2, diffY - diffY // 2])
            x = torch.cat((x, enc_feat), dim=1)  # Skip connection with dynamic padding
            x = self.up_blocks[i+1](x)
        
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        return x