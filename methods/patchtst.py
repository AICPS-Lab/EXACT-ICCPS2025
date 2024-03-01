import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, in_channels=1):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, in_channels)
        # Create patches
        seq_len = x.shape[1]
        num_patches = seq_len // self.patch_size
        x = x.unfold(1, self.patch_size, self.patch_size).contiguous()
        x = x.view(-1, x.shape[1], x.shape[-1])
        # Project patches to embed_dim
        
        x = self.proj(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return x


class PatchTST(nn.Module):
    def __init__(self, patch_size=20, embed_dims=64, num_heads=2, num_layers=2, num_classes=5, in_channels=6, dropout=0.5, input_length=300):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, embed_dims, in_channels)
        self.pos_encoder = PositionalEncoding(embed_dims)
        encoder_layers = nn.TransformerEncoderLayer(embed_dims, num_heads, embed_dims * 4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers, norm=nn.LayerNorm(embed_dims))
        num_patches = input_length // patch_size
        self.output_layer = nn.Linear(num_patches * embed_dims, input_length)
        self.num_classes_proj = nn.Linear(in_channels, num_classes)


    def forward(self, x):
        b, h, c = x.shape
        x = self.patch_embed(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.view(b, c, -1)
        x = self.output_layer(x)
        x = x.permute(0, 2, 1)
        x = self.num_classes_proj(x)
        return x
    
    def forward_pred(self, inputs):
        masks = self.forward(inputs)
        masks = masks.permute(0, 2, 1)
        probabilities = F.softmax(masks, dim=1)
        pred = torch.argmax(probabilities, dim=1)
        return pred
    
if __name__ == '__main__':
    model = PatchTST()
    print(sum(p.numel() for p in model.parameters()))
    x = torch.randn(32, 300, 6)
    print(model(x).shape)
