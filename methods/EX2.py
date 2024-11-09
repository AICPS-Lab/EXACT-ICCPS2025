import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
class ConvBlock(nn.Module):
    """A convolutional block with residual connections."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    """Attention mechanism for the skip connections."""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (from decoder)
        # x: skip connection (from encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block for capturing temporal dependencies."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        # x: [batch, channels, seq_length]
        x = x.permute(0, 2, 1)  # [batch, seq_length, channels]
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)  # [batch, channels, seq_length]
        return x

class UpConv(nn.Module):
    """Upsampling followed by a ConvBlock with attention."""
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.attention = AttentionBlock(F_g=out_channels, F_l=skip_channels, F_int=out_channels // 2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = F.interpolate(x1, size=x2.size(2), mode="linear", align_corners=False)
        x2 = self.attention(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class EX2(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--embed_dim", type=int, default=32, help="Base embedding dimension.")
        parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in the transformer.")
        parser.add_argument("--dim_feedforward", type=int, default=128, help="Dimension of the feedforward network in transformer.")
        parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in transformer.")
        parser.add_argument("--depth", type=int, default=4, help="Depth of the UNet (number of encoding layers).")
        return parser

    def __init__(self, args):
        super(EX2, self).__init__()
        embed_dim = args.embed_dim
        num_heads = args.num_heads
        dim_feedforward = args.dim_feedforward
        dropout = args.dropout
        depth = args.depth

        # Define the number of channels at each level
        self.enc_channels = [embed_dim * (2 ** i) for i in range(depth)]  # [32, 64, 128, 256]
        self.dec_channels = self.enc_channels[::-1]  # [256, 128, 64, 32]

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        for i in range(depth):
            in_ch = self.enc_channels[i -1] if i > 0 else args.in_channels
            out_ch = self.enc_channels[i]
            self.encoder_blocks.append(ConvBlock(in_ch, out_ch))
            if i < depth -1:
                self.pool_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Bottleneck
        self.bottleneck = ConvBlock(self.enc_channels[-1], self.enc_channels[-1] * 2)  # 256 -> 512
        self.transformer = TransformerEncoderBlock(
            d_model=self.enc_channels[-1] * 2,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Decoder
        self.upconvs = nn.ModuleList()
        for i in range(depth):
            j = depth - 1 - i
            in_ch = self.enc_channels[j] * 2  # e.g., 256*2=512
            out_ch = self.dec_channels[i]     # e.g., 256, 128, 64, 32
            skip_channels = self.enc_channels[j]  # e.g., 256, 128, 64, 32
            self.upconvs.append(UpConv(in_ch, out_ch, skip_channels))

        # Final Convolution
        self.final_conv = nn.Conv1d(self.dec_channels[-1], args.out_channels, kernel_size=1)  # 32 -> 2
        self.layer_norm = nn.LayerNorm(args.out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, channels, seq_length]
        encoder_features = []

        # Encoder pathway
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            encoder_features.append(x)
            if i < len(self.pool_layers):
                x = self.pool_layers[i](x)

        # Bottleneck
        x = self.bottleneck(x)
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False):
            x = self.transformer(x)

        # Decoder pathway
        for i, upconv in enumerate(self.upconvs):
            skip = encoder_features[-(i+1)]
            x = upconv(x, skip)

        out = self.final_conv(x)  # [batch, out_channels, seq_length]
        out = out.permute(0, 2, 1)  # [batch, seq_length, out_channels]
        out = self.layer_norm(out)
        return out

    def forward_pred(self, x):
        masks = self.forward(x)
        probabilities = F.softmax(masks, dim=-1)
        pred = torch.argmax(probabilities, dim=-1)
        return pred

def main():
    parser = argparse.ArgumentParser(description="EX2 Model for Multivariate Time Series")
    # Add model arguments
    EX2.add_args(parser)
    # Example additional arguments can be added here if needed
    args = parser.parse_args()

    # Instantiate the model
    model = EX2(args)
    print("Model Architecture:\n", model)

    # Example input: batch_size=32, seq_length=128, features=6
    batch_size = 8  # Adjusted to match the error traceback
    seq_length = 128
    input_features = args.in_channels
    input_tensor = torch.randn(batch_size, input_features, seq_length)  # [batch, in_channels, seq_length]

    # Forward pass
    output = model(input_tensor)  # [batch, seq_length, out_channels]
    print(f"Output shape: {output.shape}")  # Expected: [8, 128, 2]

    # Forward prediction
    predictions = model.forward_pred(input_tensor)  # [batch, seq_length]
    print(f"Predictions shape: {predictions.shape}")  # Expected: [8, 128]

if __name__ == "__main__":
    main()
