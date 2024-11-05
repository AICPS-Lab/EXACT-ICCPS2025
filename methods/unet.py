import math
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Input is BxCxD, thus, you need to concatenate along dimension 2
        x1 = F.interpolate(
            x1, size=x2.size()[2], mode="linear", align_corners=False
        )
        x = torch.cat([x2, x1], dim=1)
        # print(x.shape)
        return self.conv(x)


class UNet(nn.Module):
    @staticmethod
    def add_args(parser):
        pass
    def __init__(self, args):
        in_channels = args.in_channels
        out_channels = args.out_channels
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
    def __init__(
        self, in_channels, out_channels, cnn_embed_dims=[64, 128, 256]
    ):
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
                self.modules.append(
                    ConvBlock(cnn_embed_dims[i - 1], cnn_embed_dims[i])
                )
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


# if __name__ == '__main__':

#     # print how many parameters are in the model

#     unet = UNet(in_channels=6, out_channels=5)
#     print('Number of trainable parameters:', sum(p.numel() for p in unet.parameters() if p.requires_grad))
#     inp = torch.rand(32, 50, 6)

#     res = unet(inp)
#     print(res.shape)


import torch
import torch.nn as nn
import torch.nn.functional as F
class TemporalPositionalEncoding(nn.Module):
    def __init__(self, time_steps: int, d_model: int):
        super(TemporalPositionalEncoding, self).__init__()
        
        # Temporal Embeddings (Trainable)
        self.temporal_embeddings = nn.Parameter(torch.randn(1, time_steps, d_model) * 0.01)
        self.register_parameter('t', self.temporal_embeddings)
        
        # Positional Embeddings (Not trainable)
        self.positional_embeddings = torch.zeros(1, time_steps, d_model)
        position = torch.arange(0., time_steps).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        self.positional_embeddings[..., 0::2] = torch.sin(position * div_term)
        self.positional_embeddings[..., 1::2] = torch.cos(position * div_term)
        self.positional_embeddings = nn.Parameter(self.positional_embeddings, requires_grad=False)
        self.register_buffer('pe', self.positional_embeddings)

    def forward(self, x: torch.Tensor):
        """
        x : input data of shape [Batch, T, 6]
        """
        
        # Adding temporal embeddings
        if x.shape[1] != self.temporal_embeddings.shape[1]:
            # x shape is batch x 6x t, need to permute:
            x = x.permute(0, 2, 1)
            x = x + self.temporal_embeddings
            
            # Adding positional embeddings
            x = x + self.positional_embeddings
            # permute back to batch x t x 6
            x = x.permute(0, 2, 1)
        else:
            x = x + self.temporal_embeddings
            x = x + self.positional_embeddings
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super(ASPP, self).__init__()
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            self.aspp.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False,
                )
            )
        self.conv1d = nn.Conv1d(
            out_channels * len(dilations), in_channels, kernel_size=1
        )

    def forward(self, x):
        res = [conv(x) for conv in self.aspp]
        res = torch.cat(res, dim=1)
        return self.conv1d(res)


class EXACT_UNet(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--in_channels", type=int, default=6, help="Input channels for the model")
        parser.add_argument("--out_channels", type=int, default=2, help="Output channels for the model")
        parser.add_argument("--aspp_dim", type=int, nargs='+', default=[6, 12], help="ASPP dimensions")
        return parser
    
    
    
    def __init__(self, args):
        in_channels = args.in_channels
        out_channels = args.out_channels
        aspp_dim = args.aspp_dim
        super(EXACT_UNet, self).__init__()
        # Temporal and Positional Encoding
        self.tpe = TemporalPositionalEncoding(time_steps=200, d_model=in_channels)

        # Initial Conv Block
        self.conv_init = ConvBlock(in_channels, 12 * len(aspp_dim))

        # ASPP Block
        self.aspp = ASPP(6, 12, aspp_dim)

        # Downsampling Path
        self.conv1 = ConvBlock(12 * len(aspp_dim), 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4 = ConvBlock(256, 512)

        # Upsampling Path
        self.up1 = UpConv(512, 256)
        self.up2 = UpConv(256, 128)
        self.up3 = UpConv(128, 64)

        # Final Convolution
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to match (N, C, L)

        # Temporal and Positional Encoding
        x = self.tpe(x)
        # ASPP Block
        x = self.aspp(x)
        # Initial Conv Block
        x = self.conv_init(x)

        # Downsampling Path
        x1 = self.conv1(x)
        p1 = self.pool1(x1)

        x2 = self.conv2(p1)
        p2 = self.pool2(x2)

        x3 = self.conv3(p2)
        p3 = self.pool3(x3)

        x4 = self.conv4(p3)

        # Upsampling Path with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Final Convolution
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)  # Permute back to (N, L, C)
        return x

    def forward_pred(self, x):
        masks = self.forward(x)
        probabilities = F.softmax(
            masks, dim=2
        )  # Apply softmax along the channel dimension
        pred = torch.argmax(probabilities, dim=2)  # Get the prediction
        return pred
