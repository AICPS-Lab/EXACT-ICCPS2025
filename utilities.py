import os
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch
import pandas as pd

def get_device():
    """
    Checks for the availability of MPS and CUDA devices and returns the appropriate device.
    If neither is available, returns the CPU device.
    """
    # Check for MPS availability
    if torch.backends.mps.is_available():
        print("MPS is available.")
        return torch.device("mps")

    # Check for CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available.")
        return torch.device("cuda:1")

    
    # Check the reasons for MPS unavailability
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not built with MPS enabled.")
    elif not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
        print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device.")

    print("Neither MPS nor CUDA is available. Using CPU.")
    return torch.device("cpu")

class sliding_windows(torch.nn.Module):
    def __init__(self, width, step):
        # https://stackoverflow.com/questions/53972159/how-does-pytorchs-fold-and-unfold-work
        super(sliding_windows, self).__init__()
        self.width = width
        self.step = step

    def forward(self, input_time_series, labels):
        input_transformed = torch.swapaxes(input_time_series.unfold(-2, size=self.width, step=self.step), -2, -1)
        # For labels, we only have one dimension, so we unfold along that dimension
        labels_transformed = labels.unfold(0, self.width, self.step)
        return input_transformed, labels_transformed

    def get_num_sliding_windows(self, total_length):
        return round((total_length - (self.width - self.step)) / self.step)
    
    
def printc(text, color):
    colors = {
        "red": '\033[91m',
        "green": '\033[92m',
        "yellow": '\033[93m',
        "blue": '\033[94m',
        "magenta": '\033[95m',
        "cyan": '\033[96m',
        "white": '\033[97m',
        "black": '\033[30m',
    }
    RESET = '\033[0m'
    color_code = colors.get(color.lower(), '\033[97m')  # Default to white if color not found
    print(color_code + text + RESET)


class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        raise NotImplementedError
        super(Normalize, self).__init__()
        # Convert mean and std to tensors and reshape them to align with the last axis
        self.mean = torch.tensor(mean).view(-1, 1)
        self.std = torch.tensor(std).view(-1, 1)

    def forward(self, input):
        # check if the input has the same number of channels as the mean and std, or if mean and std is just one channel:
        if self.mean.shape[-1] != 1 and input.shape[-1] != self.mean.shape[-1]:
            raise ValueError(f"Input has {input.shape[-1]} channels, but mean has {self.mean.shape[-1]} channels.")
        # std:
        if self.std.shape[-1] != 1 and input.shape[-1] != self.std.shape[-1]:
            raise ValueError(f"Input has {input.shape[-1]} channels, but std has {self.std.shape[-1]} channels.")
        
        
        # Normalize each channel
        return (input - self.mean) / self.std
    
    
class StandardTransform(torch.nn.Module):
    def __init__(self, scaler='standard'):
        super(StandardTransform, self).__init__()
        if scaler == 'standard':
            self.scaler = StandardScaler()
        else:
            raise NotImplementedError('Only standard scaler is implemented')
        
    def __call__(self, data):
        if data.ndim == 2:
            data = self.scaler.transform(data)
        else:
            raise NotImplementedError('Only 2D data is supported')
        return torch.tensor(data, dtype=torch.float32)
    
    def fit(self, data: torch.Tensor):
        n_samples, n_time_steps, n_features = data.shape
        data_reshaped = data.reshape(-1, n_features)  # The shape becomes (n_samples * n_time_steps, n_features)
        self.scaler.fit(data_reshaped)
        printc('Fitted with mean: {}, and std: {}'.format(self.scaler.mean_, np.sqrt(self.scaler.var_)), color='red')
        return self
    
    def fit(self, files: [str]):
        # walk through the folder and load all the data into memory
        data = []
        for file in files:
            data.append(pd.read_csv(file).to_numpy()[:, 1:7])
        data = np.concatenate(data, axis=0)
        self.scaler.fit(data)
        printc('Fitted with mean: {}, and std: {}'.format(self.scaler.mean_, np.sqrt(self.scaler.var_)), color='red')
        return self
        
