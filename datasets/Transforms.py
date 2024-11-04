import numpy as np
import torch
from scipy.interpolate import CubicSpline
import random
from scipy.spatial.transform import Rotation as R


class IMUAugmentation:
    def __init__(
        self,
        rotation_chance=0.5,
        jitter_chance=0.5,
        scaling_chance=0.5,
        spline_chance=0.5,
    ):
        self.rotation_chance = rotation_chance
        self.jitter_chance = jitter_chance
        self.scaling_chance = scaling_chance
        self.spline_chance = spline_chance

    def random_quaternion(self):
        """Generate a random unit quaternion."""
        rand_nums = np.random.uniform(0, 45, size=3)
        q = R.from_euler("xyz", rand_nums, degrees=True).as_quat()
        return q

    def rotate_vector(self, v, q):
        """Rotate a vector using a quaternion."""
        rotation = R.from_quat(q)
        return rotation.apply(v)

    def apply_rotation(self, imu_data_np):
        """Apply rotation to IMU data."""
        qS_prime_S = self.random_quaternion()
        rotated_data = np.zeros(imu_data_np.shape)
        for i in range(imu_data_np.shape[0]):
            acc_data = imu_data_np[i, :3]
            gyro_data = imu_data_np[i, 3:6]
            rotated_data[i, :3] = self.rotate_vector(acc_data, qS_prime_S)
            rotated_data[i, 3:6] = self.rotate_vector(gyro_data, qS_prime_S)
        return rotated_data

    def apply_jitter(self, imu_data_np, sigma=0.01):
        """Add random jitter to IMU data."""
        jitter = np.random.normal(0, sigma, imu_data_np.shape)
        return imu_data_np + jitter

    def apply_scaling(self, imu_data_np, scale_range=(0.9, 1.1)):
        """Scale IMU data by a random factor."""
        scale_factor = np.random.uniform(*scale_range)
        return imu_data_np * scale_factor

    def apply_spline(self, imu_data_np):
        """Apply cubic spline interpolation to smooth the IMU data."""
        time_steps = np.arange(imu_data_np.shape[0])
        augmented_data = np.zeros_like(imu_data_np)
        for i in range(imu_data_np.shape[1]):
            cs = CubicSpline(time_steps, imu_data_np[:, i])
            augmented_data[:, i] = cs(time_steps)
        return augmented_data

    def __call__(self, imu_data):
        """Apply a random augmentation to 6-axis IMU data."""
        if isinstance(imu_data, torch.Tensor):
            imu_data_np = imu_data.numpy()
        elif isinstance(imu_data, np.ndarray):
            imu_data_np = imu_data
        else:
            raise TypeError("imu_data must be a torch.Tensor or np.ndarray")

        # random_number = random.random()
        if random.random() < self.rotation_chance:
            imu_data_np = self.apply_rotation(imu_data_np)
        elif random.random() < self.jitter_chance:
            imu_data_np = self.apply_jitter(imu_data_np)
        elif random.random() < self.scaling_chance:
            imu_data_np = self.apply_scaling(imu_data_np)
        elif random.random() < self.spline_chance:
            imu_data_np = self.apply_spline(imu_data_np)

        return torch.tensor(imu_data_np)
