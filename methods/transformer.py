import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, num_classes=6, ntoken=6, ninp=64, nhead=1, nhid=256, nlayers=6, dropout=0.1, init_std=.02, activation='relu'):
        super(TransformerModel, self).__init__()
        self.input_emb = nn.Linear(ntoken, ninp)
        self.ninp = ninp
        self.relu = nn.ReLU()
        # modulelist:
        encoder_layer = nn.TransformerEncoderLayer(d_model=ninp, nhead=nhead, dim_feedforward=nhid, dropout=dropout, activation=activation, batch_first=True)
        encoder_norm = nn.LayerNorm(ninp)   
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers, norm=encoder_norm)
        self.decoder = nn.Linear(ninp, num_classes)
        self.init_std = init_std
        # max layer:
        # self.max = nn.MaxPool1d(100)
        
    def init_weights(self):
        nn.init.trunc_normal_(self.input_emb.weight, std=self.init_std)
        nn.init.trunc_normal_(self.decoder.weight, std=self.init_std)
        # xaiver initialization for Transformer:
        for param in self.transformer_encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        
                    
    def forward(self, src):
        src = self.input_emb(src)
        src = self.relu(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output