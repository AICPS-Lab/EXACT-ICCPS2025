import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class ResidualConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualConvBlock1D, self).__init__()
        self.conv_block = ConvBlock1D(in_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        return F.relu(self.conv_block(x) + self.shortcut(x))


class MultiHeadSelfAttention1D(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention1D, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.pos_embed = None

    def forward(self, x):
        batch_size, channels, seq_length = x.size()
        if self.pos_embed is None or self.pos_embed.size(0) != seq_length:
            # Initialize positional encoding
            self.pos_embed = self.create_positional_encoding(
                seq_length, self.embed_dim
            ).to(x.device)

        x = x.permute(2, 0, 1)  # Shape: [S, B, E]
        x = x + self.pos_embed.unsqueeze(1)

        attn_output, _ = self.self_attn(x, x, x)
        attn_output = attn_output.permute(1, 2, 0)  # Shape: [B, E, S]
        return attn_output

    @staticmethod
    def create_positional_encoding(length, dim):
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class EX(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--embed_dim", type=int, default=32)
        parser.add_argument("--num_heads", type=int, default=4)
        return parser

    def __init__(self, args):
        embed_dim = args.embed_dim
        num_heads = args.num_heads
        in_channels = args.in_channels
        out_channels = args.out_channels
        super(EX, self).__init__()
        self.embed_dim = embed_dim

        # Encoder with Residual Connections
        self.encoder1 = ResidualConvBlock1D(in_channels, embed_dim)
        self.encoder2 = ResidualConvBlock1D(embed_dim, embed_dim * 2)
        self.encoder3 = ResidualConvBlock1D(embed_dim * 2, embed_dim * 4)
        self.encoder4 = ResidualConvBlock1D(embed_dim * 4, embed_dim)

        self.pool = nn.MaxPool1d(2, 2)

        # Self-Attention in Bottleneck
        self.mhsa = MultiHeadSelfAttention1D(embed_dim, num_heads)

        # Decoder with Residual Connections
        self.upconv4 = nn.ConvTranspose1d(
            embed_dim, embed_dim, kernel_size=2, stride=2
        )
        self.decoder4 = ResidualConvBlock1D(embed_dim + embed_dim * 4, embed_dim * 4)

        self.upconv3 = nn.ConvTranspose1d(
            embed_dim * 4, embed_dim * 4, kernel_size=2, stride=2
        )
        self.decoder3 = ResidualConvBlock1D(embed_dim * 4 + embed_dim * 2, embed_dim * 2)

        self.upconv2 = nn.ConvTranspose1d(
            embed_dim * 2, embed_dim * 2, kernel_size=2, stride=2
        )
        self.decoder2 = ResidualConvBlock1D(embed_dim * 2 + embed_dim, embed_dim)

        self.decoder1 = nn.Conv1d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, in_channels, S]
     
        # Encoder
        e1 = self.encoder1(x)                        # [B, embed_dim, S]
        e2 = self.encoder2(self.pool(e1))            # [B, embed_dim * 2, S/2]
        e3 = self.encoder3(self.pool(e2))            # [B, embed_dim * 4, S/4]
        e4 = self.encoder4(self.pool(e3))            # [B, embed_dim, S/8]

        # Self-Attention in Bottleneck
        e4 = self.mhsa(e4)                           # [B, embed_dim, S/8]

        # Decoder with Skip Connections
        d4 = self.upconv4(e4)                        # Upsample
        d4 = self._crop_and_concat(d4, e3)           # Match size and concat with e3
        d4 = self.decoder4(d4)                       # [B, embed_dim * 4, S/4]

        d3 = self.upconv3(d4)                        # Upsample
        d3 = self._crop_and_concat(d3, e2)           # Match size and concat with e2
        d3 = self.decoder3(d3)                       # [B, embed_dim * 2, S/2]

        d2 = self.upconv2(d3)                        # Upsample
        d2 = self._crop_and_concat(d2, e1)           # Match size and concat with e1
        d2 = self.decoder2(d2)                       # [B, embed_dim, S]

        # Final output layer
        d1 = self.decoder1(d2)                       # [B, out_channels, S]
        d1 = d1.permute(0, 2, 1)                     # [B, S, out_channels]
        return d1

    def _crop_and_concat(self, upsampled, bypass):
        # Crop or pad the upsampled tensor to match the size of the bypass tensor
        diff = upsampled.size(-1) - bypass.size(-1)
        if diff == 0:
            return torch.cat([upsampled, bypass], dim=1)
        elif diff > 0:
            upsampled = upsampled[..., :bypass.size(-1)]
        else:
            upsampled = F.pad(upsampled, (0, -diff))
        return torch.cat([upsampled, bypass], dim=1)

    def forward_print_shape(self, x):
        # print each layer's output shape
        x = x.permute(0, 2, 1)
        print("Input shape:", x.shape)
        e1 = self.encoder1(x)
        print("Encoder 1 shape:", e1.shape)
        e2 = self.encoder2(self.pool(e1))
        print("Encoder 2 shape:", e2.shape)
        e3 = self.encoder3(self.pool(e2))
        print("Encoder 3 shape:", e3.shape)
        e4 = self.encoder4(self.pool(e3))
        print("Encoder 4 shape:", e4.shape)
        # e4 = self.mhsa(e4)
        # print("Multi-head self-attention shape:", e4.shape)
        d4 = self.upconv4(e4)
        print("Upconv 4 shape:", d4.shape)
        d4 = self._crop_and_concat(d4, e3)
        print("Crop and concat 4 shape:", d4.shape)
        d4 = self.decoder4(d4)               
        print("Decoder 4 shape:", d4.shape)
        d3 = self.upconv3(d4)
        print("Upconv 3 shape:", d3.shape)
        d3 = self._crop_and_concat(d3, e2)
        print("Crop and concat 3 shape:", d3.shape)
        d3 = self.decoder3(d3)
        print("Decoder 3 shape:", d3.shape)
        d2 = self.upconv2(d3)
        print("Upconv 2 shape:", d2.shape)
        d2 = self._crop_and_concat(d2, e1)
        print("Crop and concat 2 shape:", d2.shape)
        d2 = self.decoder2(d2)
        print("Decoder 2 shape:", d2.shape)
        d1 = self.decoder1(d2)
        d1 = d1.permute(0, 2, 1)
        print("Output shape:", d1.shape)
        
        