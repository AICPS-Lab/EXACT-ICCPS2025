import random
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class RotationAugmentation:
    def __init__(self, random_chance=0.5):
        self.random_chance = random_chance

    def random_quaternion(self):
        """Generate a random unit quaternion."""
        rand_nums = np.random.uniform(0, 90, size=3)
        q = R.from_euler("xyz", rand_nums, degrees=True).as_quat()
        return q

    def rotate_vector(self, v, q):
        """Rotate a vector using a quaternion."""
        rotation = R.from_quat(q)
        return rotation.apply(v)

    def __call__(self, imu_data):
        """Apply rotation augmentation to the 6-axis IMU data."""
        random_number = random.random()
        if random_number <= self.random_chance:
            return imu_data
        if isinstance(imu_data, torch.Tensor):
            imu_data_np = imu_data.numpy()
        elif isinstance(imu_data, np.ndarray):
            imu_data_np = imu_data
        else:
            raise TypeError("imu_data must be a torch.Tensor or np.ndarray")
        qS_prime_S = self.random_quaternion()
        rotated_data = np.zeros(imu_data_np.shape)

        for i in range(imu_data_np.shape[0]):
            acc_data = imu_data_np[i, :3]
            gyro_data = imu_data_np[i, 3:6]

            rotated_data[i, :3] = self.rotate_vector(acc_data, qS_prime_S)
            rotated_data[i, 3:6] = self.rotate_vector(gyro_data, qS_prime_S)

        return torch.tensor(rotated_data)
