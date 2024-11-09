import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    1D CNN model for time series segmentation tasks, with a sequence of convolutional
    and pooling layers, followed by fully connected layers and a multi-class linear output layer.
    """
    def add_args(parser):
        parser.add_argument("--conv_sizes", type=int, nargs="+", default=[32, 64, 128])
        parser.add_argument("--fc_sizes", type=int, nargs="+", default=[128, 64])
        parser.add_argument("--dropout", type=float, default=0.5)
        return parser
    def __init__(self, args):
        in_channels = args.in_channels
        out_channels = args.out_channels
        fc_sizes = args.fc_sizes
        conv_sizes = args.conv_sizes
        dropout = args.dropout
        super(CNN, self).__init__()

        # Add convolutional layers
        conv_layers = []
        for conv_size in conv_sizes:
            conv_layers.append(nn.Conv1d(in_channels, conv_size, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(2))
            in_channels = conv_size
        self.conv_net = nn.Sequential(*conv_layers)

        # Compute the size of the flattened feature map
        # Assuming input size is [batch_size, in_channels, seq_len]
        self.flattened_size = in_channels  # Update if needed based on input length

        # Fully connected layers for output
        fc_layers = []
        for fc_size in fc_sizes:
            fc_layers.append(nn.Linear(self.flattened_size, fc_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            self.flattened_size = fc_size
        fc_layers.append(nn.Linear(self.flattened_size, out_channels))  # Output layer
        self.fc_net = nn.Sequential(*fc_layers)

    def forward(self, x):
        # Pass through convolutional layers
        h = self.conv_net(x)
        h = h.view(h.size(0), -1)  # Flatten for the fully connected layers
        return self.fc_net(h)