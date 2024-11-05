import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_features):
        super(CNNBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.lstm = nn.LSTM(out_channels, hidden_features, batch_first=True)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  # Change shape from (batch_size, channels, seq_len) to (batch_size, seq_len, channels)
        # x, _ = self.lstm(x)
        x = x.transpose(1, 2)  # Change shape back to (batch_size, channels, seq_len)
        x = self.bn(x)
        x = self.relu(x)

        return x

    
class DeCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(DeCNNBlock, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.act = nn.ReLU() # can be any activation function

    def forward(self, x):
        x = self.up(x)
        return self.act(x)

def l2_regularizer(scale, parameters):
    l2_reg = torch.tensor(0.).to(next(parameters).device)
    for param in parameters:
        l2_reg += torch.norm(param)
    return l2_reg * scale

# Define a model using the CNNBlock
class MyModel(nn.Module):
    def __init__(self, look_back_length=150, main_length=50, in_channels=6):
        super(MyModel, self).__init__()
        self.in_channels = in_channels
        layer1_out_channels = 8
        layer2_out_channels = 10
        # input_length = 200
        input_length = (look_back_length + main_length)//2
        self.layer1 = CNNBlock(in_channels=in_channels, out_channels=layer1_out_channels, hidden_features=input_length) # 6 x 200 => 8 x 100
        layer1_out_channels *= input_length

        input_length = input_length//2
        self.layer2 = CNNBlock(in_channels=8, out_channels=layer2_out_channels, hidden_features=input_length) # 8 x 100 => 10 x 50
        layer2_out_channels *= input_length
        
        input_length = input_length//2
        self.layer3 = CNNBlock(in_channels=10, out_channels=1, hidden_features=128) # 10 x 50 => 128x 1 => 128
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lstm = nn.LSTM(input_length, 128, batch_first=True)
        self.relinear1 = nn.Linear(layer2_out_channels, 256)
        self.relinear2 = nn.Linear(layer1_out_channels, 512)
        # Now we start the upsampling process
        self.up1 = DeCNNBlock(in_channels=1, out_channels=20)  # 1x128 => 10x50
        self.up2 = DeCNNBlock(in_channels=20, out_channels=8) # 10x50 => 8x100
        self.up3 = DeCNNBlock(in_channels=8, out_channels=1)  # 8x100 => 1x200
        self.linear = nn.Linear(1024, 50)
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        lstm_x, _ = self.lstm(x3) # Nx1x128
        x = self.up1(lstm_x) # Nx20x256
        x = self.up2(x + self.relinear1(x2.view(x2.shape[0], 1, -1)))
        x = self.up3(x + self.relinear2(x1.view(x1.shape[0], 1, -1)))
        x = torch.sigmoid(x)  # Apply sigmoid function to make output between 0 and 1
        x = x.squeeze(1)  # Remove the channel dimension
        x = self.linear(x)  # Apply linear layer to adjust the sequence length
        x = torch.sigmoid(x)
        return x

    def get_regularization_loss(self, scale):
        return l2_regularizer(scale, self.parameters())

# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
        
#         self.conv1 = nn.Conv1d(6, 16, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool1d(2)
#         self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(32 * 25, 120) # 32 * 25 assumes input_length of 50, adjust accordingly
#         self.fc2 = nn.Linear(120, 50)
        
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(x.size(0), -1) # flatten the tensor
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         x = torch.sigmoid(x)
#         return x

#     def get_regularization_loss(self, scale):
#         return l2_regularizer(scale, self.parameters())
