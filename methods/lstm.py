import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, num_classes=5, in_channels=6, embed_dims=64, num_layers=2, dropout=0.1, init_std=.02):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=embed_dims, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(embed_dims, num_classes)
        self.init_std = init_std
        self.init_weights()
        
    def init_weights(self):
        nn.init.trunc_normal_(self.decoder.weight, std=self.init_std)
        for param in self.lstm.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, src):
        output, (h_n, c_n) = self.lstm(src)
        output = self.decoder(output)
        return output
    
    def forward_pred(self, inputs):
        masks = self.forward(inputs)
        masks = masks.permute(0, 2, 1)
        probabilities = F.softmax(masks, dim=1)
        pred = torch.argmax(probabilities, dim=1)
        return pred
