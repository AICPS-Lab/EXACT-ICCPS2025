import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels//2, out_channels//2, kernel_size=3, padding=1)
        self.conv1_1 = nn.Conv1d(in_channels//2, out_channels//2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels//2, out_channels//2, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv1d(out_channels//2, out_channels//2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_acc = x[:, :x.shape[1]//2, :]
        x_gyr = x[:, x.shape[1]//2:, :]
        x_acc = self.relu(self.conv1(x_acc))
        x_acc = self.relu(self.conv2(x_acc))
        x_gyr = self.relu(self.conv1_1(x_gyr))
        x_gyr = self.relu(self.conv2_1(x_gyr))
        x = torch.cat([x_acc, x_gyr], dim=1)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels , out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is BxCxD, thus, you need to concatenate along dimension 2
        # print(x1.shape, x2.shape)
        x = torch.cat([x2, x1], dim=1)
        # print(x.shape)
        return self.conv(x)

class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet2, self).__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(128, 256)
        # self.lstm = nn.LSTM(input_size=256, hidden_size=64, batch_first=True)
        self.up1 = UpConv(256, 128)
        self.up2 = UpConv(128, 64)
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)
        module_list = []
        cnn_embed_dims = [64, 64]
        dilations = [1, 2, 4]
        for i, out_channels in enumerate(cnn_embed_dims):
            if i == 0:
                in_channels = in_channels
            else:
                in_channels = cnn_embed_dims[i - 1]
            module_list.append(
                CausalConv1D(in_channels, out_channels, kernel_size=3, dilation=dilations[i])
            )
            module_list.append(nn.BatchNorm1d(out_channels))
            module_list.append(nn.ReLU())
            module_list.append(nn.MaxPool1d(2))
        self.moduless = nn.Sequential(*module_list)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        print(self.moduless(x).shape)
        x1 = self.conv1(x)
        p1 = self.pool1(x1)
        x2 = self.conv2(p1)
        p2 = self.pool2(x2)
        x3 = self.conv3(p2)
        # lstm_out, _ = self.lstm(x3.permute(0, 2, 1))
        # lstm_out = lstm_out.permute(0, 2, 1)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)
        return x
    def forward_pred(self, x):
        masks = self.forward(x)
        masks = masks.permute(0, 2, 1)
        # probabilities = F.softmax(masks, dim=1)
        pred = torch.argmax(masks, dim=1)
        return pred
    
class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1D, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride

        # Compute the required padding to ensure causality
        self.padding = (kernel_size - 1) * dilation  # Padding on one side to ensure causality

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=0, dilation=dilation)  # No built-in padding

    def forward(self, x):
        # Manually pad the sequence on the left to ensure causality
        x_padded = nn.functional.pad(x, (self.padding, 0))  # Padding format (left, right)
        
        # Apply convolution to the padded input
        conv = self.conv1d(x_padded)
        
        return conv

# Example

    
if __name__ == '__main__':
    # print how many parameters are in the model
    
    unet = UNet2(in_channels=6, out_channels=5)
    print('Number of trainable parameters:', sum(p.numel() for p in unet.parameters() if p.requires_grad))
    inp = torch.rand(32, 300, 6)

    res = unet(inp)
    print(res.shape)