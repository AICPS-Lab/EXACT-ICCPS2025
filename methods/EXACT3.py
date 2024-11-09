import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """A convolutional block with residual connections."""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)  # Changed to inplace=False
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

class UpConv(nn.Module):
    """Upsampling followed by concatenation."""
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ensure the upsampled tensor matches the skip connection's temporal dimension
        x1 = F.interpolate(x1, size=x2.size(2), mode="linear", align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        return x

class LSTMBlock(nn.Module):
    """LSTM layer to capture temporal dependencies."""
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first
        )
        self.relu = nn.ReLU(inplace=False)  # Changed to inplace=False

    def forward(self, x):
        # x shape: [batch, channels, seq_length]
        x = x.permute(0, 2, 1)  # [batch, seq_length, channels]
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # [batch, channels, seq_length]
        return x

class LSTMUNet(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--embed_dim", type=int, default=32, help="Base embedding dimension.")
        parser.add_argument("--lstm_hidden_size", type=int, default=32, help="Hidden size for LSTM layers.")
        parser.add_argument("--lstm_num_layers", type=int, default=1, help="Number of layers in LSTM.")
        parser.add_argument("--bidirectional", action='store_true', help="Use bidirectional LSTM.")
        parser.add_argument("--depth", type=int, default=4, help="Depth of the UNet (number of encoding layers).")
        return parser

    def __init__(self, args):
        super(LSTMUNet, self).__init__()
        embed_dim = args.embed_dim
        lstm_hidden_size = args.lstm_hidden_size
        lstm_num_layers = args.lstm_num_layers
        bidirectional = args.bidirectional
        depth = args.depth
        in_channels = args.in_channels
        out_channels = args.out_channels

        # Define the number of channels at each level
        self.enc_channels = [embed_dim * (2 ** i) for i in range(depth)]  # e.g., [32, 64, 128, 256]
        self.dec_channels = self.enc_channels[::-1]  # e.g., [256, 128, 64, 32]

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.lstm_encoders = nn.ModuleList()
        for i in range(depth):
            in_ch = self.enc_channels[i -1] if i > 0 else in_channels
            out_ch = self.enc_channels[i]
            self.encoder_blocks.append(ConvBlock(in_ch, out_ch))
            # Set LSTM hidden_size to match ConvBlock's out_channels
            self.lstm_encoders.append(LSTMBlock(
                input_size=out_ch,
                hidden_size=out_ch,  # Ensures channel consistency
                num_layers=lstm_num_layers,
                bidirectional=bidirectional
            ))
            if i < depth -1:
                self.pool_layers.append(nn.MaxPool1d(kernel_size=2, stride=2))

        # Bottleneck
        self.bottleneck = ConvBlock(self.enc_channels[-1], self.enc_channels[-1] * 2)
        self.lstm_bottleneck = LSTMBlock(
            input_size=self.enc_channels[-1] * 2,
            hidden_size=self.enc_channels[-1] * 2,  # Ensures channel consistency
            num_layers=lstm_num_layers,
            bidirectional=bidirectional
        )

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.lstm_decoders = nn.ModuleList()
        for i in range(depth):
            j = depth - 1 - i
            in_ch = self.dec_channels[i] * 2  # After concatenation with skip connection
            out_ch = self.dec_channels[i]
            self.upconvs.append(UpConv(in_ch, out_ch))
            self.decoder_blocks.append(ConvBlock(out_ch * 2, out_ch))
            self.lstm_decoders.append(LSTMBlock(
                input_size=out_ch,
                hidden_size=out_ch,  # Ensures channel consistency
                num_layers=lstm_num_layers,
                bidirectional=bidirectional
            ))

        # Final Convolution
        self.final_conv = nn.Conv1d(self.dec_channels[-1], out_channels, kernel_size=1)
        self.layer_norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, channels, seq_length]
        encoder_features = []

        # Encoder pathway
        for i in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[i](x)
            x = self.lstm_encoders[i](x)
            encoder_features.append(x)
            if i < len(self.pool_layers):
                x = self.pool_layers[i](x)

        # Bottleneck
        x = self.bottleneck(x)
        x = self.lstm_bottleneck(x)

        # Decoder pathway
        for i in range(len(self.upconvs)):
            skip = encoder_features[-(i+1)]
            x = self.upconvs[i](x, skip)
            x = self.decoder_blocks[i](x)
            x = self.lstm_decoders[i](x)

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
    parser = argparse.ArgumentParser(description="LSTM-UNet Model for Multivariate Time Series")
    # Add model arguments
    LSTMUNet.add_args(parser)
    # Example additional arguments can be added here if needed
    args = parser.parse_args()

    # Instantiate the model
    model = LSTMUNet(args)
    print("Model Architecture:\n", model)

    # Example input: batch_size=8, seq_length=128, features=6
    batch_size = 8
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
