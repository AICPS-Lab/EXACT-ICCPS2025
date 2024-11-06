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

def sort_filename(files, order=-1):
    """
    Sort the filenames in the list in ascending order.
    """
    first_order = order
    # return the file order expected to be 0, 1, 2, 3; but not 0, 10, 11, 12, 2, 3 following by all the other things:
    return sorted(files, key=lambda x: ("_".join(x[0].split(".")[0].split("_")[0:first_order]),
                                        int(x[0].split(".")[0].split("/")[-1].split("_")[first_order])))


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


import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_patterned_imu_noise(
    data,
    reference_length=5,
    max_length=25,
    noise_type="static_pause",
    range_percentage=0.05,
    peak_intensity=1.0,
    peak_probability=0.1,
    directional_bias=0.1,
):
    # Separate accelerometer and gyroscope data from the initial pattern
    accel_baseline = data[-reference_length:, 0:3]
    gyro_baseline = data[-reference_length:, 3:6]

    # Calculate the mean and range for static pause fluctuation level
    accel_mean = accel_baseline.mean(axis=0)
    gyro_mean = gyro_baseline.mean(axis=0)
    accel_range = accel_baseline.max(axis=0) - accel_baseline.min(axis=0)
    gyro_range = gyro_baseline.max(axis=0) - gyro_baseline.min(axis=0)

    # Set fluctuation level based on a percentage of the range
    fluctuation_level_accel = range_percentage * accel_range
    fluctuation_level_gyro = range_percentage * gyro_range

    # Number of time steps in the noise segment
    T = np.random.randint(max_length // 5, max_length)

    # Generate directional trend
    directional_trend = (
        directional_bias * np.linspace(0, 1, T).reshape(-1, 1)
        * np.random.choice([-1, 1], size=(1, 3))
    )

    # Generate random fluctuations for accelerometer and gyroscope for each time step
    accel_fluctuations = np.random.uniform(
        -fluctuation_level_accel, fluctuation_level_accel, (T, 3)
    )
    gyro_fluctuations = np.random.uniform(
        -fluctuation_level_gyro, fluctuation_level_gyro, (T, 3)
    )

    # Determine peak indices for accelerometer and gyroscope randomly
    peak_indices = np.random.rand(T) < peak_probability
    peak_accel = np.random.uniform(-peak_intensity, peak_intensity, (T, 3))
    peak_gyro = np.random.uniform(-peak_intensity, peak_intensity, (T, 3))

    # Apply peak intensity only at selected indices
    accel_fluctuations[peak_indices] = peak_accel[peak_indices]
    gyro_fluctuations[peak_indices] = peak_gyro[peak_indices]

    # Combine trend and fluctuation based on noise type
    if noise_type == "directional_shift":
        accel_samples = accel_mean + directional_trend + accel_fluctuations
        gyro_samples = gyro_mean + directional_trend + gyro_fluctuations
    else:
        accel_samples = accel_mean + accel_fluctuations
        gyro_samples = gyro_mean + gyro_fluctuations

    # Generate random rotation vectors and apply them to the accelerometer data
    random_axes = np.random.uniform(-1, 1, (T, 3))
    random_axes /= np.linalg.norm(random_axes, axis=1).reshape(-1, 1)
    rotation_angles = np.random.uniform(
        -fluctuation_level_accel.mean(), fluctuation_level_accel.mean(), T
    )
    rotation_vectors = random_axes * rotation_angles.reshape(-1, 1)
    rotations = R.from_rotvec(rotation_vectors)
    rotated_accel = rotations.apply(accel_samples)

    # Stack accelerometer and gyroscope data to form the IMU noise segment
    imu_noise_segment = np.hstack([rotated_accel, gyro_samples])

    return imu_noise_segment


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
    imu_noise_segment = generate_patterned_imu_noise(
        data,
        reference_length=reference_length,
        noise_type="static_pause",
        max_length=noise_shape[0],
        range_percentage=0.5,
        peak_probability=0
    )

    return imu_noise_segment


def idle_movement(noise, reference_length, noise_shape):
    """
    Generate idle movement noise with more variability and slight directional trends.

    Parameters:
        noise (array): Base noise array to reference recent sensor values.
        reference_length (int): Number of rows to calculate baseline mean.
        noise_shape (tuple): Shape of the generated idle movement noise (T, D), where T is time steps, D is the number of IMU axes.

    Returns:
        array: Generated idle movement noise segment.
    """
    imu_noise_segment = generate_patterned_imu_noise(
        noise,
        reference_length=reference_length,
        noise_type="directional_shift",
        max_length=noise_shape[0],
        range_percentage=0.5,
        peak_probability=0)

    return imu_noise_segment


def incident_movement(
    noise, reference_length, noise_shape, directional_bias=0.1
):
    raise NotImplementedError


def environmental_movement(
    noise, reference_length, noise_shape, directional_bias=0.1
):
    raise NotImplementedError


def generate_sudden_change(noise, noise_length):
    """
    Generate a sudden change or quick peak noise segment.

    Parameters:
        noise (array): Base noise array to reference recent sensor values.
        noise_length (int): Length of the generated noise segment.
        intensity (float): Multiplier to control the magnitude of the peak.
        peak_duration (int): Duration of the peak within the noise_shape.

    Returns:
        array: Generated sudden change noise segment.
    """
    imu_noise_segment = generate_patterned_imu_noise(
        noise,
        reference_length=5,
        noise_type="sudden_change",
        max_length=noise_length,
        range_percentage=0.5, # rest position
        peak_probability=.5,
        peak_intensity=0.3,
    )
    return imu_noise_segment


def generate_nonexercise(max_length=50):
    
    pickle_filename = './datasets/OpportunityUCIDataset/loco_2_mask.npy'
    data = np.load(pickle_filename, allow_pickle=True)
    data = data.item()
    inp = data['inputs']/ 9.98
    labels = data['labels']
    #randomly select a segment of size max_length only from labels 0, 1, 2,3 (not 4 as iti is lying down)
    indices = np.where(labels < 4)[0]
    start = np.random.choice(indices)
    end = start + np.random.randint(max_length // 5, max_length)
    return inp[start:end, :]