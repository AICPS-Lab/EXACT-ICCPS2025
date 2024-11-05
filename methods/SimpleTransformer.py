import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, look_back_length=150, main_length=50, in_channels=6, d_model=128, nhead=8, num_layers=3):
        super(Transformer, self).__init__()

        # BiLSTM layer for encoding
        self.bilstm = nn.LSTM(in_channels, d_model // 2, num_layers=1, bidirectional=True)

        # Transformer layers
        self.transformer = nn.Transformer(d_model, nhead, num_layers)

        # Fully connected layer for decoding
        self.decoder = nn.Linear(d_model, main_length)

    def forward(self, x):
        # Input shape: (B, T, Z)

        # Rearrange to shape (T, B, Z) for LSTM
        x = x.permute(1, 0, 2)

        # BiLSTM encoding
        x, _ = self.bilstm(x)
        
        # Note: The output x now has shape (T, B, d_model)
        # because it's a concatenation of hidden states from both directions
        
        # Transformer forward pass
        x = self.transformer(x, x)
        
        # Take the output corresponding to the first token in the sequence
        x = x[0]
        
        # Decoder to output shape (B, M)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        
        return x