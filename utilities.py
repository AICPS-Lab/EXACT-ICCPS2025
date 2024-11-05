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


import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_patterned_imu_noise(
    data,
    reference_length=5,
    max_length=25,
    noise_type="static_pause",
    range_percentage=0.05,
    peak_intensity=1.0,  # Fixed peak intensity
    peak_probability=0.1,
    directional_bias=0.1,
):
    """
    Generate IMU noise that follows the pattern of initial data, with optional idle movements and directional trends.

    Parameters:
        initial_data (array): Initial set of values representing current position.
        max_length (int): Length of the generated noise segment.
        noise_type (str): Type of movement to simulate ("static_pause", "sudden_change", or "directional_shift").
        range_percentage (float): Percentage of the data range to set fluctuation level for static pauses.
        peak_intensity (float): Fixed intensity for sudden peaks.
        peak_probability (float): Probability of a peak occurring within the noise.
        directional_bias (float): Bias to simulate a gradual directional trend.

    Returns:
        array: Combined accelerometer and gyroscope noise segment following the initial pattern.
    """
    # Separate accelerometer and gyroscope data from the initial pattern
    accel_baseline = data[-reference_length:, 0:3]
    gyro_baseline = data[-reference_length:, 3:6]

    # Calculate the mean and range for static pause fluctuation level
    accel_mean = accel_baseline.mean(axis=0)
    gyro_mean = gyro_baseline.mean(axis=0)
    accel_range = accel_baseline.max(axis=0) - accel_baseline.min(axis=0)
    gyro_range = gyro_baseline.max(axis=0) - gyro_baseline.min(axis=0)

    # Set fluctuation level based on a percentage of the range (for static pause)
    fluctuation_level_accel = range_percentage * accel_range
    fluctuation_level_gyro = range_percentage * gyro_range

    # Initialize storage for generated accelerometer and gyroscope noise
    accel_data = []
    gyro_data = []
    current_orientation = R.from_euler("xyz", [0, 0, 0]).as_quat()

    # Generate a directional trend over time
    T = np.random.randint(max_length // 5, max_length)
    directional_trend = (
        directional_bias
        * np.linspace(0, 1, T).reshape(-1, 1)
        * np.random.choice([-1, 1], size=(1, 3))
    )
    counter = (0, -1, -1)  # number of times, acc-axis, gyro-axis
    for t in range(T):
        if (
            noise_type == "sudden_change"
            and np.random.rand() < peak_probability
            or (counter[0] < 5 and counter[1] >= 0)
        ):
            # Generate a single, high-intensity peak and not using uniform but generate ONE or TWO random value for each axis different from the original data:
            # and if peaks happen in that axis it should continue to have for a few time steps
            if counter[1] < 0:
                counter = (counter[0], np.random.randint(0, 3), counter[2])
            if counter[2] < 0:
                counter = (counter[0], counter[1], np.random.randint(0, 3))
            accel_fluctuation = np.random.uniform(
                -fluctuation_level_accel, fluctuation_level_accel
            )
            gyro_fluctuation = np.random.uniform(
                -fluctuation_level_gyro, fluctuation_level_gyro
            )
            counter = (counter[0] + 1, counter[1], counter[2])
            accel_fluctuation[counter[1]] = np.random.uniform(
                -peak_intensity, peak_intensity
            )
            gyro_fluctuation[counter[2]] = np.random.uniform(
                -peak_intensity, peak_intensity
            )

            # Apply the peak fluctuation to baseline mean
            accel_sample = accel_mean + accel_fluctuation
            gyro_sample = gyro_mean + gyro_fluctuation
            if counter[0] == 5:
                counter = (0, -1, -1)

        elif noise_type == "directional_shift":
            # Directional shift with small fluctuations plus directional trend
            accel_fluctuation = np.random.uniform(
                -fluctuation_level_accel, fluctuation_level_accel
            )
            gyro_fluctuation = np.random.uniform(
                -fluctuation_level_gyro, fluctuation_level_gyro
            )

            # Add the directional trend to the baseline mean
            accel_sample = (
                accel_mean + directional_trend[t, :] + accel_fluctuation
            )
            gyro_sample = gyro_mean + directional_trend[t, :] + gyro_fluctuation

        else:  # Static pause or default case with small fluctuations
            accel_fluctuation = np.random.uniform(
                -fluctuation_level_accel, fluctuation_level_accel
            )
            gyro_fluctuation = np.random.uniform(
                -fluctuation_level_gyro, fluctuation_level_gyro
            )

            # Apply fluctuations around the baseline without any directional trend
            accel_sample = accel_mean + accel_fluctuation
            gyro_sample = gyro_mean + gyro_fluctuation

        # Quaternion rotation for accelerometer data
        random_axis = np.random.uniform(-1, 1, 3)
        random_axis /= np.linalg.norm(random_axis)  # Normalize the axis
        rotation_angle = np.random.uniform(
            -fluctuation_level_accel.mean(), fluctuation_level_accel.mean()
        )
        rotation_quat = R.from_rotvec(rotation_angle * random_axis).as_quat()
        accel_orientation = R.from_quat(current_orientation) * R.from_quat(
            rotation_quat
        )
        rotated_accel = accel_orientation.apply(accel_sample)

        # Append the generated data
        accel_data.append(rotated_accel)
        gyro_data.append(gyro_sample)

        # Update the current orientation for continuous rotation
        current_orientation = accel_orientation.as_quat()

    # Convert to arrays and stack accelerometer and gyroscope data
    accel_data = np.array(accel_data)
    gyro_data = np.array(gyro_data)
    imu_noise_segment = np.hstack([accel_data, gyro_data])

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
