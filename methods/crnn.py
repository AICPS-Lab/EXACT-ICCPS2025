import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, in_channels=6, num_classes=5, cnn_embed_dims=[64, 64, 64, 64], embed_dims=50):
        super(CRNN, self).__init__()
        if isinstance(cnn_embed_dims, int):
            cnn_embed_dims = [cnn_embed_dims]
        module_list = []
        for i, out_channels in enumerate(cnn_embed_dims):
            if i == 0:
                in_channels = in_channels
            else:
                in_channels = cnn_embed_dims[i - 1]
            module_list.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            module_list.append(nn.BatchNorm1d(out_channels))
            module_list.append(nn.ReLU())
            module_list.append(nn.MaxPool1d(2))
        self.lstm = nn.LSTM(input_size=64, hidden_size=embed_dims, batch_first=True)
        self.fc = nn.Linear(embed_dims, num_classes)
        self.cnn = nn.Sequential(*module_list)
        self.init_weights()
        
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        b, h, c = x.shape
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        # upsampling:
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=h, mode='linear')
        x = x.permute(0, 2, 1)
        return x
    
    def forward_pred(self, x):
        masks = self.forward(x)
        masks = masks.permute(0, 2, 1)
        probabilities = F.softmax(masks, dim=1)
        pred = torch.argmax(probabilities, dim=1)
        return pred