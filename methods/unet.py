import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(128, 256)
        self.up1 = UpConv(256, 128)
        self.up2 = UpConv(128, 64)
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x1 = self.conv1(x)
        p1 = self.pool1(x1)
        x2 = self.conv2(p1)
        p2 = self.pool2(x2)
        x3 = self.conv3(p2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)
        return x
    def forward_pred(self, x):
        masks = self.forward(x)
        masks = masks.permute(0, 2, 1)
        probabilities = F.softmax(masks, dim=1)
        pred = torch.argmax(probabilities, dim=1)
        return pred

class UNet_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, cnn_embed_dims=[64, 128, 256]):
        super(UNet_encoder, self).__init__()
        # self.conv1 = ConvBlock(in_channels, 64)
        # self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.conv2 = ConvBlock(64, 128)
        # self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.conv3 = ConvBlock(128, 256)
        # self.final_conv = nn.Linear(256, out_channels)
        self.modules = []

        for i in range(len(cnn_embed_dims)):
            if i == 0:
                self.modules.append(ConvBlock(in_channels, cnn_embed_dims[i]))
            else:
                self.modules.append(ConvBlock(cnn_embed_dims[i-1], cnn_embed_dims[i]))
            self.modules.append(nn.MaxPool1d(kernel_size=2, stride=2))
        self.final_conv = nn.Linear(cnn_embed_dims[-1], out_channels)

        self.backbone = nn.Sequential(*self.modules)
        


    def forward(self, x):
        x = x.permute(0, 2, 1)
        x1 = self.backbone(x)
        x3_maxpooled = F.max_pool1d(x1, kernel_size=x1.size(2)).squeeze()
        out = self.final_conv(x3_maxpooled)
        return out
    
    def forward_pred(self, x):
        masks = self.forward(x)
        probabilities = F.softmax(masks, dim=1)
        pred = torch.argmax(probabilities, dim=1)
        return pred
    
if __name__ == '__main__':

    # print how many parameters are in the model
    
    unet = UNet(in_channels=6, out_channels=5)
    print('Number of trainable parameters:', sum(p.numel() for p in unet.parameters() if p.requires_grad))
    inp = torch.rand(32, 50, 6)

    res = unet(inp)
    print(res.shape)