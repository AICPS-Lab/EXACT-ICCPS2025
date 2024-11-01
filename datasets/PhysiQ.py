import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utilities import seed
from .QueryDataset import QueryDataset
import os
import argparse

DEFAULT_INFO = "data is a dictionary with keys as subject and values as list of data files, label is a dictionary with keys as subject and values as list of labels, label_to_exer_rom is a list of tuples with each tuple as (exercise, rom)"


class PhysiQ(QueryDataset):
    # TODO: for testing we should also have per subject fsl

    DATASET_NAME = "physiq"
    NUM_TASKS = 16

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
        super(PhysiQ, self).__init__(
            root, split, window_size, window_step, bg_fg, args, transforms
        )

    def _process_data(self):
        folders = [
            "segment_sessions_one_repetition_data_E1",
            "segment_sessions_one_repetition_data_E2",
            "segment_sessions_one_repetition_data_E3",
            "segment_sessions_one_repetition_data_E4",
        ]
        data_folder = os.path.join(self.root, self.DATASET_NAME)
        # physiq dataset 4 exercises and each contains variation (rom), each variation will be one class
        data = []
        subject = []
        subjLabel_to_data = {}
        unique_indices = []
        for folder in folders:
            dirs = os.path.join(data_folder, folder)
            # walk through the directory
            # print(dirs)
            for root, _, files in os.walk(dirs):
                for file in files:
                    # file is csv:
                    if file.endswith(".csv"):
                        data.append(os.path.join(root, file))
                        subject.append(file.split("_")[0])
                        ind_label = (file.split("_")[1], file.split("_")[3])
                        if (
                            file.split("_")[0],
                            file.split("_")[1],
                            file.split("_")[3],
                        ) not in subjLabel_to_data:
                            subjLabel_to_data[
                                (
                                    file.split("_")[0],
                                    file.split("_")[1],
                                    file.split("_")[3],
                                )
                            ] = []
                        subjLabel_to_data[
                            (
                                file.split("_")[0],
                                file.split("_")[1],
                                file.split("_")[3],
                            )
                        ].append(os.path.join(root, file))
                        if ind_label not in unique_indices:
                            unique_indices.append(ind_label)
        # print("subject: ", subject)
        # print("Indices: ", unique_indices)
        seed(self.args.dataset_seed)
        train = {}
        test = {}
        train_data = {}
        train_label = {}
        test_data = {}
        test_label = {}

        for k, v in subjLabel_to_data.items():
            # randomly sample 20 percent of the data for testing:
            random_indices = random.sample(range(len(v)), int(0.2 * len(v)))
            subj = (k[0], k[1])  # NOTE: k[0]
            ind_label = (k[1], k[2])
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
        train["data"] = train_data
        train["label"] = train_label
        train["label_to_exer_rom"] = unique_indices
        train["info"] = DEFAULT_INFO

        test["data"] = test_data
        test["label"] = test_label
        test["label_to_exer_rom"] = unique_indices
        test["info"] = DEFAULT_INFO
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

        return

    @staticmethod
    def add_args(parser):
        """Add dataset-specific arguments to the parser."""
        parser.add_argument(
            "--shuffle",
            type=str,
            default="random",
            help="Shuffle the data on each subject",
        )
        parser.add_argument(
            "--dataset_seed",
            type=int,
            default=42,
            help="Seed for shuffling the data",
        )
        return parser

    def load_data(self, split):
        data = np.load(
            os.path.join(self.root, self.DATASET_NAME, f"{split}_data.npy"),
            allow_pickle=True,
        ).item()
        return data

    def concanetate_data(self):
        seed(self.args.dataset_seed)
        data = self.data["data"]
        label = self.data["label"]
        label_to_exer_rom = self.data["label_to_exer_rom"]
        exercise_label = []
        for i in range(len(label_to_exer_rom)):
            if label_to_exer_rom[i][0] not in exercise_label:
                exercise_label.append(label_to_exer_rom[i][0])
        res_data = []
        res_label = []
        res_exer_label = []
        for k, v in data.items():
            file = []
            dense_label = []
            k_label = label[k]
            if self.args.shuffle == "random":
                combined = list(zip(v, k_label))
                random.shuffle(combined)
            else:
                combined = list(zip(v, k_label))

            for ind, (filename, original_label) in enumerate(combined):
                df_np = pd.read_csv(filename).to_numpy()[:, 1:7]
                file.append(df_np)

                if self.bg_fg is not None:
                    pass
                    # dense_label.append([1 if original_label == self.bg_fg else 0] * df_np.shape[0])
                else:
                    dense_label.append([original_label] * df_np.shape[0])

            file = np.concatenate(file, axis=0)
            dense_label = np.concatenate(dense_label, axis=0)

            sfile, sdense_label = self.sw(
                torch.tensor(file), torch.tensor(dense_label)
            )
            res_data.append(sfile)
            res_label.append(sdense_label)
            res_exer_label.append(
                torch.tensor(
                    [exercise_label.index(label_to_exer_rom[original_label][0])]
                    * sfile.shape[0]
                )
            )
        res_data = torch.cat(res_data, axis=0)
        res_label = torch.cat(res_label, axis=0)

        res_exer_label = torch.cat(res_exer_label, axis=0)

        return res_data, res_label, res_exer_label

    def if_npy_exists(self, split):
        return os.path.exists(
            os.path.join(self.root, self.DATASET_NAME, f"{split}_data.npy")
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transforms is not None:
            return (
                self.transforms(self.data[idx]),
                self.label[idx],
                self.res_exer_label[idx],
            )
        # print(self.data[idx].shape, self.transforms(self.data[idx]).shape)
        
        return self.data[idx], self.label[idx], self.res_exer_label[idx]
