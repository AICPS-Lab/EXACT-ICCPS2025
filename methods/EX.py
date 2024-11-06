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


class ChannelAttention1D(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(
            in_planes, in_planes // ratio, kernel_size=1, bias=False
        )
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(
            in_planes // ratio, in_planes, kernel_size=1, bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, channels, seq_length]
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        scale = self.sigmoid(out)
        return x * scale


class AttentionConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionConvBlock1D, self).__init__()
        self.conv_block = ConvBlock1D(in_channels, out_channels)
        self.channel_attention = ChannelAttention1D(out_channels)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.channel_attention(x)
        return x


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


class MultiHeadCrossAttention1D(nn.Module):
    def __init__(self, query_dim, key_dim, embed_dim, num_heads):
        super(MultiHeadCrossAttention1D, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(key_dim, embed_dim)
        self.value_proj = nn.Linear(key_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x, cross_input):
        batch_size, query_channels, seq_length = x.size()
        batch_size, key_channels, seq_length = cross_input.size()

        x = x.permute(2, 0, 1)  # Shape: [S, B, query_channels]
        cross_input = cross_input.permute(
            2, 0, 1
        )  # Shape: [S, B, key_channels]

        # Project to embedding dimension
        query = self.query_proj(x)  # [S, B, embed_dim]
        key = self.key_proj(cross_input)  # [S, B, embed_dim]
        value = self.value_proj(cross_input)  # [S, B, embed_dim]

        attn_output, _ = self.cross_attn(query, key, value)
        attn_output = attn_output.permute(1, 2, 0)  # [B, embed_dim, S]
        return attn_output


class EX(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument("--embed_dim", type=int, default=128)
        parser.add_argument("--num_heads", type=int, default=8)
        return parser
    def __init__(self, args):
        embed_dim = args.embed_dim
        num_heads = args.num_heads
        in_channels = args.in_channels
        out_channels = args.out_channels
        super(EX, self).__init__()
        self.embed_dim = embed_dim

        # Encoder with Channel Attention
        self.encoder1 = AttentionConvBlock1D(in_channels, embed_dim)
        self.encoder2 = AttentionConvBlock1D(embed_dim, embed_dim *2)
        self.encoder3 = AttentionConvBlock1D(embed_dim *2, embed_dim *4)
        self.encoder4 = AttentionConvBlock1D(embed_dim *4, embed_dim)

        self.pool = nn.MaxPool1d(2, 2)

        # Attention modules
        self.mhsa = MultiHeadSelfAttention1D(embed_dim, num_heads)
        self.mhca4 = MultiHeadCrossAttention1D(
            embed_dim, embed_dim *4, embed_dim, num_heads
        )
        self.mhca3 = MultiHeadCrossAttention1D(embed_dim *4, embed_dim *2, embed_dim, num_heads)
        self.mhca2 = MultiHeadCrossAttention1D(embed_dim *2, embed_dim, embed_dim, num_heads)

        # Decoder with Channel Attention
        self.decoder4 = AttentionConvBlock1D(self.embed_dim + embed_dim *4, embed_dim *4)
        self.decoder3 = AttentionConvBlock1D(embed_dim * 2 + embed_dim * 2, embed_dim * 2)
# embed_dim * 2 + embed_dim * 2 = 128 + 128 = 256
        self.decoder2 = AttentionConvBlock1D(embed_dim *2 + embed_dim, embed_dim)
        self.decoder1 = nn.Conv1d(embed_dim, out_channels, kernel_size=1)

        # Upsampling layers
        self.upconv4 = nn.ConvTranspose1d(
            embed_dim, embed_dim, kernel_size=2, stride=2
        )
        self.upconv3 = nn.ConvTranspose1d(embed_dim*4, embed_dim*2, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose1d(embed_dim*2, embed_dim*2, kernel_size=2, stride=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, in_channels, S]
 
        # Encoder
        e1 = self.encoder1(x)                        # [B, 64, S]
        e2 = self.encoder2(self.pool(e1))            # [B, 128, S/2]

        e3 = self.encoder3(self.pool(e2))            # [B, 256, S/4]
        e4 = self.encoder4(self.pool(e3))            # [B, embed_dim, S/8]

        # Self-Attention in Bottleneck (optional)
        e4 = self.mhsa(e4)                           # [B, embed_dim, S/8]

        # Decoder with Skip Connections
        d4 = self.upconv4(e4)                        # [B, embed_dim, S/4]
        d4 = torch.cat([d4, e3], dim=1)              # [B, embed_dim + 256, S/4]

        d4 = self.decoder4(d4)                       # [B, 256, S/4]

        d3 = self.upconv3(d4)                        # [B, 256, S/2]
        d3 = torch.cat([d3, e2], dim=1)              # [B, 256 + 128, S/2]
        d3 = self.decoder3(d3)                       # [B, 128, S/2]

        d2 = self.upconv2(d3)                        # [B, 128, S]
        d2 = torch.cat([d2, e1], dim=1)              # [B, 128 + 64, S]
        d2 = self.decoder2(d2)                       # [B, 64, S]

        # Final output layer
        d1 = self.decoder1(d2)                       # [B, out_channels, S]
        d1 = d1.permute(0, 2, 1)                     # [B, S, out_channels]
        return d1
