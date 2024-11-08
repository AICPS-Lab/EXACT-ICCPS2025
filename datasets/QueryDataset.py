# QueryDataset.py
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utilities import *


class QueryDataset(Dataset):
    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        window_size: int = 300,
        window_step: int = 50,
        bg_fg=None,
        args=None,
        transforms=None,
        test_subject: int = None,
    ):
        assert split in [
            "train",
            "test",
        ], "Split must be either 'train' or 'test'"
        assert not (
            args.loocv and test_subject is None
        ), "Test subject must be specified for LOOCV"
        assert (
            isinstance(test_subject, int) or test_subject is None
        ), "Test subject must be an integer if specified"
        self.root = root
        self.bg_fg = bg_fg
        self.split = split
        self.window_size = window_size
        self.window_step = window_step
        self.args = args
        self.transforms = transforms
        self.test_subject = test_subject
        self.loocv = args.loocv
        self.DATASET_NAME = self.get_dataset_name()

        self.sw = sliding_windows(window_size, window_step)

        if not self.if_npy_exists():
            self._process_data()
        if self.loocv and not self.if_npy_exists(loocv=True):
            self._process_loocv_data()

        self.data_dict = self.load_data()
        self.data, self.label, self.res_exer_label = self.concatenate_data()

    def get_dataset_name(self):
        """
        Should return the name of the dataset.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def default_data_folders(self):
        """
        Returns the default data folder(s) for the dataset.
        Can be overridden by subclasses.
        """
        return [os.path.join(self.root, self.DATASET_NAME)]

    def parse_filename(self, filename):
        """
        Parses the filename and returns a dictionary with keys:
        'subject', 'ind_label', 'unique_identifier'
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_subject(self):
        """
        Returns the index of the subject in the filename.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_ind_label(self):
        """
        Returns the index of the label in the filename.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _process_data(self):
        data_folders = self.default_data_folders()
        subjLabel_to_data = {}
        unique_indices = []

        for data_folder in data_folders:
            for root_dir, _, files in os.walk(data_folder):
                for file in files:
                    if file.endswith(".csv"):
                        filepath = os.path.join(root_dir, file)
                        identifiers = self.parse_filename(file)
                        if identifiers is None:
                            continue  # Skip files that do not match expected pattern

                        subj = identifiers["subject"]
                        ind_label = identifiers["ind_label"]
                        unique_identifier = identifiers["unique_identifier"]

                        if unique_identifier not in subjLabel_to_data:
                            subjLabel_to_data[unique_identifier] = []
                        subjLabel_to_data[unique_identifier].append(filepath)

                        if ind_label not in unique_indices:
                            unique_indices.append(ind_label)
        unique_indices = sorted(
            unique_indices, key=lambda x: (int(x[0][1:]), x[1])
        )

        # Proceed with splitting data into train/test and saving
        seed(self.args.dataset_seed)
        train_data = {}
        train_label = {}
        test_data = {}
        test_label = {}
        for k, v in subjLabel_to_data.items():
            random_indices = random.sample(range(len(v)), int(0.2 * len(v)))
            subj = tuple([k[i] for i in self.get_subject()])
            ind_label = tuple([k[i] for i in self.get_ind_label()])
            if subj not in train_data:
                train_data[subj] = []
                train_label[subj] = []
                test_data[subj] = []
                test_label[subj] = []
            train_data[subj].extend(
                [v[i] for i in range(len(v)) if i not in random_indices]
            )
            train_label[subj].extend(
                [
                    unique_indices.index(ind_label)
                    for i in range(len(v))
                    if i not in random_indices
                ]
            )
            test_data[subj].extend(
                [v[i] for i in range(len(v)) if i in random_indices]
            )
            test_label[subj].extend(
                [
                    unique_indices.index(ind_label)
                    for i in range(len(v))
                    if i in random_indices
                ]
            )

        train = {
            "data": train_data,
            "label": train_label,
            "unique_indices": unique_indices,
            "info": "Processed data",
        }

        test = {
            "data": test_data,
            "label": test_label,
            "unique_indices": unique_indices,
            "info": "Processed data",
        }

        # Save processed data
        np.save(
            os.path.join(self.root, self.DATASET_NAME, "train_data.npy"),
            train,
            allow_pickle=True,
        )
        np.save(
            os.path.join(self.root, self.DATASET_NAME, "test_data.npy"),
            test,
            allow_pickle=True,
        )

    def _process_loocv_data(self):
        loocv_data_path = os.path.join(
            self.root, self.DATASET_NAME, "loocv_data.npy"
        )
        if os.path.exists(loocv_data_path):
            print("LOOCV data already processed. Skipping processing.")
            return  # Data already processed

        data_folders = self.default_data_folders()
        subjLabel_to_data = {}
        unique_indices = []

        for data_folder in data_folders:
            for root_dir, _, files in os.walk(data_folder):
                for file in files:
                    if file.endswith(".csv"):
                        filepath = os.path.join(root_dir, file)
                        identifiers = self.parse_filename(file)
                        if identifiers is None:
                            continue

                        subj = identifiers["subject"]
                        ind_label = identifiers["ind_label"]
                        unique_identifier = identifiers["unique_identifier"]

                        if unique_identifier not in subjLabel_to_data:
                            subjLabel_to_data[unique_identifier] = []
                        subjLabel_to_data[unique_identifier].append(filepath)

                        if ind_label not in unique_indices:
                            unique_indices.append(ind_label)
        unique_indices = sorted(
            unique_indices, key=lambda x: (int(x[0][1:]), x[1])
        )
        # Save the processed data
        loocv_data = {
            "subjLabel_to_data": subjLabel_to_data,
            "unique_indices": unique_indices,
            "info": "LOOCV processed data",
        }
        np.save(loocv_data_path, loocv_data, allow_pickle=True)

    def if_npy_exists(self, loocv=False):
        if loocv:
            return os.path.exists(
                os.path.join(self.root, self.DATASET_NAME, "loocv_data.npy")
            )
        else:
            return all(
                os.path.exists(
                    os.path.join(
                        self.root, self.DATASET_NAME, f"{split}_data.npy"
                    )
                )
                for split in ["train", "test"]
            )

    def load_data(self):
        if self.loocv:
            data = np.load(
                os.path.join(self.root, self.DATASET_NAME, "loocv_data.npy"),
                allow_pickle=True,
            ).item()
        else:
            data = np.load(
                os.path.join(
                    self.root, self.DATASET_NAME, f"{self.split}_data.npy"
                ),
                allow_pickle=True,
            ).item()
        return data

    def concatenate_data(self):
        seed(self.args.dataset_seed)
        if self.loocv:
            return self._loocv_concatenate_data()
        else:
            return self._normal_concatenate_data()

    def _normal_concatenate_data(self):
        data = self.data_dict["data"]
        label = self.data_dict["label"]
        unique_indices = self.data_dict["unique_indices"]
        exercise_labels = sorted(set([label[0] for label in unique_indices]))
        print(unique_indices)
        print(exercise_labels)
        res_data = []
        res_label = []
        res_exer_label = []
        for k, v in data.items():
            file = []
            dense_label = []
            k_label = label[k]
            combined = list(zip(v, k_label))

            if self.args.shuffle == "random":
                random.shuffle(combined)
            elif self.args.shuffle == "sorted":
                combined = sort_filename(combined)
            else:
                raise NotImplementedError

            for filename, original_label in combined:
                df_np = pd.read_csv(filename).to_numpy()[:, 1:7]
                file.append(df_np)
                dense_label.append([original_label] * df_np.shape[0])

                if self.args.add_side_noise:
                    noise = self.generate_noise(df_np, self.args.noise_type)
                    file.append(noise)
                    dense_label.append([-1] * noise.shape[0])

            file = np.concatenate(file, axis=0)
            dense_label = np.concatenate(dense_label, axis=0)

            if self.args.add_side_noise:
                dense_label = dense_label + 1

            sfile, sdense_label = self.sw(
                torch.tensor(file), torch.tensor(dense_label)
            )
            res_data.append(sfile)
            res_label.append(sdense_label)
            res_exer_label.append(
                torch.tensor(
                    [exercise_labels.index(unique_indices[original_label][0])]
                    * sfile.shape[0]
                )
            )

        res_data = torch.cat(res_data, axis=0)
        res_label = torch.cat(res_label, axis=0)
        res_exer_label = torch.cat(res_exer_label, axis=0)
        return res_data, res_label, res_exer_label

    def _loocv_concatenate_data(self):
        data_dict = self.data_dict
        subjLabel_to_data = data_dict["subjLabel_to_data"]
        unique_indices = data_dict["unique_indices"]

        if self.test_subject is None:
            raise ValueError("test_subject must be specified for LOOCV")
        unique_test_subjects = sorted(
            set([key[0] for key in subjLabel_to_data.keys()]),
            key=lambda x: int(x[1::]),
        )

        if self.test_subject not in range(1, len(unique_test_subjects) + 1):
            raise ValueError(
                f"Test subject {self.test_subject} not found in dataset {unique_test_subjects}."
            )

        # Directory to save splits for reproducibility
        split_dir = os.path.join(
            self.root, self.DATASET_NAME, "loocv_splits", str(self.test_subject)
        )
        train_split_file = os.path.join(split_dir, "train_data.npy")
        test_split_file = os.path.join(split_dir, "test_data.npy")

        if os.path.exists(train_split_file) and os.path.exists(test_split_file):
            # Load existing splits
            train_data = np.load(train_split_file, allow_pickle=True).item()
            test_data = np.load(test_split_file, allow_pickle=True).item()
        else:
            # Create splits and save them
            train_data = {}
            test_data = {}

            for key, files in subjLabel_to_data.items():
                subj = tuple([key[i] for i in self.get_subject()])
                ind_label = tuple([key[i] for i in self.get_ind_label()])
                if unique_test_subjects.index(subj[0]) == self.test_subject:
                    if subj not in test_data:
                        test_data[key] = []
                    test_data[key].extend(files)
                else:
                    if key not in train_data:
                        train_data[key] = []
                    train_data[key].extend(files)
            # Create directory if it doesn't exist
            os.makedirs(split_dir, exist_ok=True)
            # Save splits
            np.save(train_split_file, train_data, allow_pickle=True)
            np.save(test_split_file, test_data, allow_pickle=True)

        # Select data based on the split
        if self.split == "train":
            data_to_use = train_data
        elif self.split == "test":
            data_to_use = test_data
        else:
            raise ValueError("Invalid split")
        # Process data_to_use
        res_data = []
        res_label = []
        res_exer_label = []
        exercise_labels = sorted(set([label[0] for label in unique_indices]))
        for key, files in data_to_use.items():
            subj = tuple([key[i] for i in self.get_subject()])
            ind_label = tuple([key[i] for i in self.get_ind_label()])
            original_label = unique_indices.index(tuple(ind_label))
            combined = [(f, original_label) for f in files]

            if self.args.shuffle == "random":
                random.shuffle(combined)
            elif self.args.shuffle == "sorted":
                combined = sort_filename(combined)
            else:
                raise NotImplementedError

            file = []
            dense_label = []
            for filename, original_label in combined:
                df_np = pd.read_csv(filename).to_numpy()[:, 1:7]
                file.append(df_np)
                dense_label.append([original_label] * df_np.shape[0])

                if self.args.add_side_noise:
                    if random.random() < 0.5:
                        noise = self.generate_noise(df_np, self.args.noise_type)
                        file.append(noise)
                        dense_label.append([-1] * noise.shape[0])

            file = np.concatenate(file, axis=0)
            dense_label = np.concatenate(dense_label, axis=0)

            if self.args.add_side_noise:
                dense_label = dense_label + 1

            sfile, sdense_label = self.sw(
                torch.tensor(file), torch.tensor(dense_label)
            )
            res_data.append(sfile)
            res_label.append(sdense_label)
            res_exer_label.append(
                torch.tensor(
                    [exercise_labels.index(ind_label[0])] * sfile.shape[0]
                )
            )

        res_data = torch.cat(res_data, axis=0)
        res_label = torch.cat(res_label, axis=0)
        res_exer_label = torch.cat(res_exer_label, axis=0)

        return res_data, res_label, res_exer_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transforms is not None:
            return (
                self.transforms(self.data[idx]),
                self.label[idx],
                self.res_exer_label[idx],
            )
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
            noise = generate_nonexercise(
                max_length * 3
            )  # NOTE: Should be longer than max_length
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
