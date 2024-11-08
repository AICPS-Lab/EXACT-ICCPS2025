import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Interpolate to match x2's length if necessary
        if x1.size(2) != x2.size(2):
            x1 = F.interpolate(
                x1, size=x2.size(2), mode="linear", align_corners=False
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, time_steps: int, d_model: int):
        super(TemporalPositionalEncoding, self).__init__()

        # Temporal Embeddings (Trainable)
        self.temporal_embeddings = nn.Parameter(
            torch.randn(1, d_model, time_steps) * 0.01
        )
        self.register_parameter("t", self.temporal_embeddings)

        # Positional Embeddings (Not trainable)
        self.positional_embeddings = torch.zeros(1, d_model, time_steps)
        position = torch.arange(0.0, time_steps).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )  # [C//2]

        # Compute sin and cos terms
        sin_term = torch.sin(position * div_term)  # [T, C//2]
        cos_term = torch.cos(position * div_term)  # [T, C//2]

        # Reshape to [1, C//2, T] for assignment
        sin_term = sin_term.transpose(0, 1).unsqueeze(0)  # [1, C//2, T]
        cos_term = cos_term.transpose(0, 1).unsqueeze(0)  # [1, C//2, T]

        # Assign to positional_embeddings
        self.positional_embeddings[:, 0::2, :] = sin_term
        self.positional_embeddings[:, 1::2, :] = cos_term

        # Register as a buffer to avoid being considered as a parameter
        self.positional_embeddings = nn.Parameter(
            self.positional_embeddings, requires_grad=False
        )
        self.register_buffer("pe", self.positional_embeddings)

    def forward(self, x: torch.Tensor):
        """
        x : input data of shape [Batch, C, T]
        """
        # Adding temporal and positional embeddings
        if x.shape[2] != self.temporal_embeddings.shape[2]:
            raise ValueError(
                f"Input time steps ({x.shape[2]}) do not match temporal embeddings ({self.temporal_embeddings.shape[2]})."
            )
        x = x + self.temporal_embeddings  # [B, C, T]
        x = x + self.positional_embeddings  # [B, C, T]
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
        parser.add_argument(
            "--aspp_dim",
            type=int,
            nargs="+",
            default=[6, 12],
            help="ASPP dilation rates",
        )
        parser.add_argument(
            "--latent_dim",
            type=int,
            default=128,
            help="Number of latent channels in the bottleneck",
        )
        return parser

    def __init__(self, args):
        in_channels = args.in_channels
        out_channels = args.out_channels
        aspp_dilations = args.aspp_dim
        window_size = args.window_size
        latent_dim = args.latent_dim  # New parameter
        super(EXACT_UNet, self).__init__()

        # Define scaling factors based on latent_dim
        # Adjust these factors as needed to suit your architecture
        self.scale_init = latent_dim // 8   # e.g., 64 if latent_dim=512
        self.scale1 = latent_dim // 4        # e.g., 128
        self.scale2 = latent_dim // 2        # e.g., 256
        self.scale3 = latent_dim             # e.g., 512

        # Initial Conv Block
        self.conv_init = ConvBlock(in_channels, self.scale_init)

        # Temporal and Positional Encoding
        self.tpe = TemporalPositionalEncoding(
            time_steps=window_size, d_model=self.scale_init
        )

        # ASPP Block
        self.aspp = ASPP(self.scale_init, self.scale_init, aspp_dilations)

        # Downsampling Path
        self.conv1 = ConvBlock(self.scale_init, self.scale1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = ConvBlock(self.scale1, self.scale2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(self.scale2, self.scale3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottleneck with latent_dim
        self.conv4 = ConvBlock(self.scale3, latent_dim)

        # Upsampling Path
        self.up1 = UpConv(latent_dim, self.scale3)
        self.up2 = UpConv(self.scale3, self.scale2)
        self.up3 = UpConv(self.scale2, self.scale1)

        # Final Convolution
        self.final_conv = nn.Conv1d(self.scale1, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Permute to match (N, C, L)

        # Initial Conv Block
        x = self.conv_init(x)     # [Batch, Channels=64, Time]

        # Temporal and Positional Encoding
        x = self.tpe(x)           # [Batch, Channels=64, Time]

        # ASPP Block
        x = self.aspp(x)          # [Batch, Channels=64, Time]

        # Downsampling Path
        x1 = self.conv1(x)
        p1 = self.pool1(x1)

        x2 = self.conv2(p1)
        p2 = self.pool2(x2)

        x3 = self.conv3(p2)
        p3 = self.pool3(x3)

        # Bottleneck
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
