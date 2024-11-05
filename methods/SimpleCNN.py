import torch
import torch.nn as nn
import torch.nn.functional as F
from .ASPP import ASPP
from .embedding import TemporalPositionalEmbedding
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (B, C, T)
        B, C, T = x.size()
        # Query
        q = self.query_conv(x)
        # Key
        k = self.key_conv(x)
        # Value
        v = self.value_conv(x)
        # Perform attention
        attn = torch.bmm(q.permute(0, 2, 1), k)  # (B, T, C) x (B, C, T) -> (B, T, T)
        attn = self.softmax(attn)
        
        y = torch.bmm(v, attn.permute(0, 2, 1))  # (B, C, T) x (B, T, T) -> (B, C, T)
        
        return y

    
class SimpleCNN(nn.Module):
    def __init__(self, look_back_length=150, main_length=50, in_channels=6, aspp=False, dilations=[1, 5, 15, 25, 50]):
        super(SimpleCNN, self).__init__()
        self.in_channels = in_channels
        self.aspp = aspp
        self.hidden1 = 64
        self.self_attention = SelfAttention(128)
        self.tpe = TemporalPositionalEmbedding(time_steps=look_back_length + main_length, d_model=in_channels)
        if aspp:
            self.aspp_block = ASPP(in_channels, self.hidden1, kernel_size=3, dilations=dilations)
            self.conv1 = nn.Conv1d(self.hidden1 * len(dilations), self.hidden1, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = nn.Conv1d(in_channels, self.hidden1, kernel_size=3, stride=1, padding=1)

        # dropout:
        self.dropout = nn.Dropout(p=0.5)

        self.conv2 = nn.Conv1d(self.hidden1, 128, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, main_length)

        

    def forward(self, x):
        # Input shape: (B, T, Z)
        # x = x.permute(0, 2, 1)  # Rearrange to (B, Z, T) for 1D CNN
        x = x.permute(0, 2, 1)
        x = self.tpe(x)

        if self.aspp:
            x = self.aspp_block(x)
            # print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.self_attention(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
        