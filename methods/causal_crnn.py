import torch
import torch.nn as nn
from torch.nn import functional as F
class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1D, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride

        # Compute the required padding to ensure causality
        self.padding = (kernel_size - 1) * dilation  # Padding on one side to ensure causality

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size,
                                stride=stride, padding=0, dilation=dilation)  # No built-in padding

    def forward(self, x):
        # Manually pad the sequence on the left to ensure causality
        x_padded = nn.functional.pad(x, (self.padding, 0))  # Padding format (left, right)
        
        # Apply convolution to the padded input
        conv = self.conv1d(x_padded)
        
        return conv



class CCRNN(nn.Module):
    # causal CRNN with dilated convolutions:
    def __init__(self, in_channels=6, num_classes=5, cnn_embed_dims=[64, 64], embed_dims=50, dilations=[1, 2, 4, 8]):
        super(CCRNN, self).__init__()
        if isinstance(cnn_embed_dims, int):
            cnn_embed_dims = [cnn_embed_dims]
        module_list = []
        for i, out_channels in enumerate(cnn_embed_dims):
            if i == 0:
                in_channels = in_channels
            else:
                in_channels = cnn_embed_dims[i - 1]
            module_list.append(
                CausalConv1D(in_channels, out_channels, kernel_size=3, dilation=dilations[i])
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