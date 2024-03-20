import copy
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels , out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is BxCxD, thus, you need to concatenate along dimension 2
        x1 = F.interpolate(x1, size=x2.size()[2], mode='linear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        # print(x.shape)
        return self.conv(x)

class TransformerBlock(nn.Module):
    def __init__(self, ninp=64, num_heads=1, embed_dims=256, dropout=0.1, init_std=.02, activation='relu'):
        super(TransformerBlock, self).__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=ninp, 
                                               nhead=num_heads, 
                                               dim_feedforward=embed_dims, 
                                               dropout=dropout, 
                                               activation=activation,
                                               batch_first=True)
        self.layerNorm = nn.LayerNorm(ninp)
    def forward(self, x):
        x = self.layerNorm(x)
        x = self.layer(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, in_channels=6, ninp=32, num_layers=9, extract_layers=[3, 6, 9]):
        super().__init__()
        self.embeddings = nn.Linear(in_channels, ninp)
        self.layer = nn.ModuleList()
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(ninp=ninp)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers

class UNetrt(nn.Module):
    def __init__(self, in_channels, out_channels,):
        super(UNetrt, self).__init__()
        self.t = Transformer()
        self.conv0 = ConvBlock(in_channels, 32)
        self.pool0 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = ConvBlock(32, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(32, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(32, 256)
        self.up1 = UpConv(256, 128)
        self.up2 = UpConv(128, 64)
        self.up3 = UpConv(64, 32)
        self.final_conv = nn.Conv1d(32, out_channels, kernel_size=1)

    def forward(self, x):
        z3, z6, z9 = self.t(x)
        # print(len(res), res[0].shape, res[1].shape, res[2].shape)
        x = x.permute(0, 2, 1)
        
        x0 = self.conv0(x)
        x1 = self.conv1(z3.permute(0, 2, 1))
        p1 = self.pool1(x1)
        x2 = self.conv2(z6.permute(0, 2, 1))
        p2 = self.pool2(x2)
        x3 = self.conv3(z9.permute(0, 2, 1))
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.up3(x, x0)
        
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)
        return x
    def forward_pred(self, x):
        masks = self.forward(x)
        masks = masks.permute(0, 2, 1)
        probabilities = F.softmax(masks, dim=1)
        pred = torch.argmax(probabilities, dim=1)
        return pred
    

if __name__ == "__main__":
    model = UNetrt(in_channels=6, out_channels=5)
    x = torch.randn(1, 200, 6)
    y = model(x)
    print(y.shape)
    pred = model.forward_pred(x)
    print(pred.shape)
    print(pred)