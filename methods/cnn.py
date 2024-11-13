import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    1D CNN model for time series segmentation tasks, with a sequence of convolutional
    and pooling layers, followed by a final convolution layer to produce predictions
    across the window size for each output channel.
    """

    def add_args(parser):
        parser.add_argument(
            "--conv_sizes", type=int, nargs="+", default=[32, 64, 128]
        )
        parser.add_argument(
            "--dropout", type=float, default=0.5
        )
        return parser

    def __init__(self, args):
        super(CNN, self).__init__()
        in_channels = args.in_channels
        out_channels = args.out_channels
        conv_sizes = args.conv_sizes
        dropout = args.dropout

        # Add convolutional layers
        conv_layers = []
        for conv_size in conv_sizes:
            conv_layers.append(
                nn.Conv1d(in_channels, conv_size, kernel_size=3, padding=1)
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool1d(2))
            conv_layers.append(nn.Dropout(dropout))
            in_channels = conv_size
        self.conv_net = nn.Sequential(*conv_layers)

        # Final convolutional layer to produce predictions for each timestep
        self.final_conv = nn.Conv1d(conv_sizes[-1], out_channels, kernel_size=1)

    def forward(self, x):
        # Permute for (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)
        h = self.conv_net(x)  # Pass through convolutional layers
        
        # Upsample to match the original sequence length if needed
        h = F.interpolate(h, size=x.size(2), mode='linear', align_corners=False)
        
        # Apply the final convolution to get output of shape (batch_size, out_channels, window_size)
        h = self.final_conv(h)

        # Permute to (batch_size, window_size, out_channels) as required
        return h.permute(0, 2, 1)
