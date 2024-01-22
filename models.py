import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[..., 0::2] = torch.sin(position * div_term)
        pe[..., 1::2] = torch.cos(position * div_term)

        if batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]

        return self.dropout(x)


class TransformerModel(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, ntoken=6, ninp=512, nhead=8, nhid=2048, nlayers=6, dropout=0.5):
        super(TransformerModel, self).__init__(d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers, batch_first=True)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout, batch_first=True)

        self.input_emb = nn.Linear(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, 1)
        # self.psi = nn.Linear(100, 5)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz)))

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        
        src = self.input_emb(src) # embedding
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask) # transformer encoder
        output = self.decoder(output)
        
        # additional for classifi:
        # output = self.psi(output.squeeze(-1))
        # return F.log_softmax(output, dim=-1)
    
        return F.sigmoid(output).squeeze(-1) # return F.log_softmax(output, dim=-1)
    
    
    
class CNNModel(nn.Module):
    """ classification model"""
    def __init__(self, ntoken=6, ninp=512, nhead=8, nhid=2048, nlayers=6, dropout=0.5):
        super(CNNModel, self).__init__()
        self.model_type = 'CNN'
        self.input_emb = nn.Linear(ntoken, ninp)
        self.ninp = ninp
        self.conv1 = nn.Conv1d(ninp, 64, 3)
        self.conv2 = nn.Conv1d(64, 128, 3)
        self.conv3 = nn.Conv1d(128, 256, 3)
        self.conv4 = nn.Conv1d(256, 512, 3)
        self.conv5 = nn.Conv1d(512, 1024, 3)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.fc2.bias)
        nn.init.uniform_(self.fc2.weight, -initrange, initrange)

    def forward(self, src):
        src = self.input_emb(src) # embedding
        src = src.transpose(1, 2)
        src = self.pool(F.relu(self.conv1(src)))
        src = self.pool(F.relu(self.conv2(src)))
        src = self.pool(F.relu(self.conv3(src)))
        src = self.pool(F.relu(self.conv4(src)))
        src = self.pool(F.relu(self.conv5(src)))
        src = src.view(-1, 1024)
        src = F.relu(self.fc1(src))
        src = self.dropout(src)
        output = self.fc2(src)
        return F.sigmoid(output).squeeze(-1)
    