import os
import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch
import pandas as pd


def seed(sd=0, cudnn=False, deterministic=False):
    np.random.seed(sd)
    random.seed(sd)
    torch.manual_seed(sd)
    torch.backends.cudnn.benchmark = cudnn
    torch.cuda.manual_seed(sd)
    torch.use_deterministic_algorithms(deterministic)


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
        return torch.device("cuda")

    # Check the reasons for MPS unavailability
    if not torch.backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch install was not built with MPS enabled."
        )
    elif not (
        torch.backends.mps.is_available() and torch.backends.mps.is_built()
    ):
        print(
            "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device."
        )

    print("Neither MPS nor CUDA is available. Using CPU.")
    return torch.device("cpu")


class sliding_windows(torch.nn.Module):
    def __init__(self, width, step):
        super(sliding_windows, self).__init__()
        self.width = width
        self.step = step

    def forward(self, input_time_series, labels):
        # Calculate number of sliding windows
        num_windows = self.get_num_sliding_windows(input_time_series.size(-2))

        # Calculate the required total length to fit exact sliding windows
        required_length = num_windows * self.step + self.width - self.step
        total_length = input_time_series.size(-2)

        # If needed, pad the input_time_series by repeating the initial segment
        if total_length < required_length:
            num_pads = required_length - total_length
            if num_pads > 0:
                padding = input_time_series[:num_pads]  # Pad from the beginning
                input_time_series = torch.cat(
                    (input_time_series, padding), dim=0
                )

        # Create sliding windows for input
        input_transformed = torch.swapaxes(
            input_time_series.unfold(-2, size=self.width, step=self.step),
            -2,
            -1,
        )

        # Handle labels similarly
        if labels is not None:
            total_labels_length = labels.size(0)
            if total_labels_length < required_length:
                num_pads = required_length - total_labels_length
                if num_pads > 0:
                    padding_labels = labels[:num_pads]
                    labels = torch.cat((labels, padding_labels), dim=0)
            labels_transformed = labels.unfold(0, self.width, self.step)
        else:
            labels_transformed = None

        return input_transformed, labels_transformed

    def get_num_sliding_windows(self, total_length):
        return max(
            1, round((total_length - (self.width - self.step)) / self.step)
        )


def printc(*args, color="red"):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "black": "\033[30m",
    }
    RESET = "\033[0m"
    color_code = colors.get(
        color.lower(), "\033[97m"
    )  # Default to white if color not found

    # Convert all arguments to string and join them with a space
    text = " ".join(map(str, args))
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
            raise ValueError(
                f"Input has {input.shape[-1]} channels, but mean has {self.mean.shape[-1]} channels."
            )
        # std:
        if self.std.shape[-1] != 1 and input.shape[-1] != self.std.shape[-1]:
            raise ValueError(
                f"Input has {input.shape[-1]} channels, but std has {self.std.shape[-1]} channels."
            )

        # Normalize each channel
        return (input - self.mean) / self.std


class StandardTransform(torch.nn.Module):
    def __init__(self, scaler="standard"):
        super(StandardTransform, self).__init__()
        if scaler == "standard":
            self.scaler = StandardScaler()
        else:
            raise NotImplementedError("Only standard scaler is implemented")

    def __call__(self, data):
        if data.ndim == 2:
            data = self.scaler.transform(data)
        else:
            raise NotImplementedError("Only 2D data is supported")
        return torch.tensor(data, dtype=torch.float32)

    def fit(self, data: torch.Tensor):
        n_samples, n_time_steps, n_features = data.shape
        data_reshaped = data.reshape(
            -1, n_features
        )  # The shape becomes (n_samples * n_time_steps, n_features)
        self.scaler.fit(data_reshaped)
        printc(
            "Fitted with mean: {}, and std: {}".format(
                self.scaler.mean_, np.sqrt(self.scaler.var_)
            ),
            color="red",
        )
        return self

    def fit(self, files: [str]):
        # walk through the folder and load all the data into memory
        data = []
        for file in files:
            data.append(pd.read_csv(file).to_numpy()[:, 1:7])
        data = np.concatenate(data, axis=0)
        self.scaler.fit(data)
        printc(
            "Fitted with mean: {}, and std: {}".format(
                self.scaler.mean_, np.sqrt(self.scaler.var_)
            ),
            color="red",
        )
        return self


def static_pause(data, reference_length, noise_shape):
    """
    Generate static pause noise based on a reference range from the data.

    Parameters:
        data (array): Base noise data to extract min and max values.
        reference_length (int): Number of rows from the end of data to calculate min and max.
        noise_shape (tuple): Shape of the new noise array.

    Returns:
        array: Generated static pause noise.
    """
    # Calculate baseline min and max from the last `reference_length` rows
    baseline_min = data[-reference_length:, :].min(axis=0, keepdims=True)
    baseline_max = data[-reference_length:, :].max(axis=0, keepdims=True)

    # Generate uniform fluctuations within the baseline range
    fluctuation = np.random.uniform(baseline_min, baseline_max, noise_shape)
    baseline_mean = data[-reference_length:, :].mean(axis=0, keepdims=True)
    noise = np.full(noise_shape, baseline_mean) + fluctuation

    return noise


def idle_movement(noise, reference_length, noise_shape, directional_bias=0.1):
    """
    Generate idle movement noise with more variability and slight directional trends.
    
    Parameters:
        noise (array): Base noise array to reference recent sensor values.
        reference_length (int): Number of rows to calculate baseline mean.
        noise_shape (tuple): Shape of the generated idle movement noise (T, D), where T is time steps, D is the number of IMU axes.
        
    Returns:
        array: Generated idle movement noise segment.
    """
    if noise_shape is None:
        raise ValueError("noise_shape must be provided to generate idle movement")

    # Calculate baseline from the last few rows of noise
    baseline_mean = noise[-reference_length:, :].mean(axis=0, keepdims=True)
    

    # Define a slightly larger fluctuation range for idle movements
    fluctuation_range = 0.1  # Larger than static pause for more noticeable idle movements
    fluctuation = np.random.uniform(
        -fluctuation_range, fluctuation_range, noise_shape
    )

    # Add a small directional trend to simulate a shift (e.g., leaning or adjusting posture)
    trend = np.linspace(0, directional_bias, noise_shape[0]).reshape(-1, 1) * np.random.choice([-1, 1], size=(1, noise.shape[1]))
    idle_movement = np.full(noise_shape, baseline_mean) + fluctuation + trend

    return idle_movement


def incident_movement(noise, reference_length, noise_shape, directional_bias=0.1):
    raise NotImplementedError

def environmental_movement(noise, reference_length, noise_shape, directional_bias=0.1):
    raise NotImplementedError

def generate_sudden_change(noise, noise_shape, intensity=2.0, peak_duration=5):
    """
    Generate a sudden change or quick peak noise segment.
    
    Parameters:
        noise (array): Base noise array to reference recent sensor values.
        noise_shape (tuple): Desired shape of the generated sudden change segment.
        intensity (float): Multiplier to control the magnitude of the peak.
        peak_duration (int): Duration of the peak within the noise_shape.
        
    Returns:
        array: Generated sudden change noise segment.
    """
    # Calculate baseline from the last few rows for smooth transition
    baseline_mean = noise[-5:, :].mean(axis=0, keepdims=True)

    # Initialize the sudden change segment with baseline values
    sudden_change = np.full(noise_shape, baseline_mean)

    # Determine the start of the peak within the segment
    peak_start = np.random.randint(0, noise_shape[0] - peak_duration)

    # Generate a sharp, sudden peak in a random direction on each axis
    peak_values = intensity * np.random.uniform(-1, 1, (peak_duration, noise_shape[1]))

    # Insert the peak into the sudden change segment
    sudden_change[peak_start:peak_start + peak_duration, :] += peak_values

    return sudden_change
