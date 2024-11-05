import torch
import torch.nn as nn
import torch.nn.functional as F
class ASPP(nn.Module):
    # dynamic sliding windows
    def __init__(self, in_channels, out_channels, kernel_size, dilations):
        super(ASPP, self).__init__()
        
        self.aspp_blocks = nn.ModuleList()
        
        for dilation in dilations:
            self.aspp_blocks.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=dilation)
            )
        
    def forward(self, x):
        out = []
        for aspp in self.aspp_blocks:
            out.append(aspp(x))
        return torch.cat(out, dim=1)  # Concatenating along the channels dimension
    

# make a functional version of ASPP:
def aspp(x, Ws, Bs, dilations):
    out = []
    
    for i, dilation in enumerate(dilations):
        out.append(
            F.conv1d(x, Ws[i], Bs[i], dilation=dilation, padding=dilation)
        )
    res = torch.cat(out, dim=1) # Concatenating along the channels dimension
    return res 