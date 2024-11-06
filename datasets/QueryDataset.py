from torch.utils.data import Dataset

from utilities import sliding_windows
from utilities import *


class QueryDataset(Dataset):
    ## time series data, spliting dataset in support and query set
    ## always 2-way N-shot 2 ways are 0 and 1, 0 is the all the other classes, 1 is the class of interest
    ## support set is the set of examples used to learn the model
    ## query set is the set of examples used to test the model
    ## N-shot is the number of examples used to learn the model
    ## in here, N-shot represents from the same 'class' or exercises
    def __init__(
        self,
        root="data",
        split="train",
        window_size=300,
        window_step=50,
        bg_fg=None,
        args=None,
        transforms=None,
    ):
        self.root = root
        # if bg_fg is not None:
        #     if N_way != 2:
        #         Warning("N_way is set to 2, because bg_fg is set to True")
        #     N_way = 2
        # self.N_way = N_way
        assert split in ["train", "test"], f"Invalid split: {split}"
        self.bg_fg = bg_fg
        self.split = split
        self.window_size = window_size
        self.window_step = window_step
        self.args = args
        self.transforms = transforms

        self.sw = sliding_windows(window_size, window_step)
        if not self.if_npy_exists(split):
            self._process_data()

        self.data = self.load_data(split)
        self.data, self.label, self.res_exer_label = self.concanetate_data()

    def concanetate_data(self):
        raise NotImplementedError

    def _process_data(self):
        raise NotImplementedError

    def load_data(self, split):
        raise NotImplementedError

    def if_npy_exists(self, split):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.res_exer_label[idx]

    def generate_noise(
        self,
        noise,
        noise_type,
        max_length=50,
        reference_length=10,
        mu=0,
        sigma=0.05,
    ):
        """
        Generate noise with given shape
        The length of the noise can vary by selecting a random length up to max_length.

        Parameters:
            noise (array): Base noise array to take reference values from.
            max_length (int): Maximum possible length of the generated noise.
            reference_length (int): Number of rows used to calculate the baseline min and max.
            mu (float): Mean for white noise.
            sigma (float): Standard deviation for white noise.
        """
        # Define the shape of the new noise array with a random length up to max_length
        noise_shape = (
            np.random.randint(
                max_length // 5, max_length + 1
            ),  # Random length between 1 and max_length
            noise.shape[1],
        )

        # Generate noise based on the selected noise type
        if noise_type == "white":
            # Generate white noise with normal distribution
            noise = np.random.normal(mu, sigma, noise_shape)
        elif noise_type == "static":
            # Generate static pause noise using the last reference_length rows
            noise = static_pause(noise, reference_length, noise_shape)
        elif noise_type == "idle":
            noise = idle_movement(noise, reference_length, noise_shape)
        elif noise_type == "sudden":
            noise = generate_sudden_change(noise, max_length)  # max_length
        elif noise_type == "nonexercise":
            noise = generate_nonexercise(max_length)
        elif noise_type == "all":
            # randomly pick one:
            noise_type = random.choice(
                ["static", "idle", "sudden", "nonexercise"]
            )
            noise = self.generate_noise(
                noise, noise_type, max_length, reference_length, mu, sigma
            )
        else:
            raise NotImplementedError(
                f"Noise type {noise_type} is not implemented"
            )
        return noise
