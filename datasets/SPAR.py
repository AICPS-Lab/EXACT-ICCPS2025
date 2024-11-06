import random
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utilities import *
from .QueryDataset import QueryDataset
import os
import argparse

DEFAULT_INFO = "data is a dictionary with keys as subject and values as list of data files, label is a dictionary with keys as subject and values as list of labels, label_to_exer_hand is a list of tuples with each tuple as (exercise, hand)"


class SPAR(QueryDataset):
    DATASET_NAME = "spar"

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
        super(SPAR, self).__init__(
            root, split, window_size, window_step, bg_fg, args, transforms
        )

    def _process_data(self):
        data_folder = os.path.join(self.root, "spar", "spar_dataset")
        data = []
        subject = []
        subjLabel_to_data = {}
        unique_indices = []
        for file in os.listdir(data_folder):
            if file.endswith(".csv"):
                data.append(os.path.join(data_folder, file))
                subject.append(file.split("_")[0])
                ind_label = (
                    file.split("_")[1],
                    file.split("_")[2],
                )  # unique label (exercise, left-right hand)
                unique_identifier = (
                    file.split("_")[0],
                    file.split("_")[1],
                    file.split("_")[2],
                )
                if unique_identifier not in subjLabel_to_data:
                    subjLabel_to_data[unique_identifier] = []
                subjLabel_to_data[unique_identifier].append(
                    os.path.join(data_folder, file)
                )
                if ind_label not in unique_indices:
                    unique_indices.append(ind_label)

        seed(self.args.dataset_seed)
        train = {}
        test = {}
        train_data = {}
        train_label = {}
        test_data = {}
        test_label = {}
        for k, v in subjLabel_to_data.items():
            random_indices = random.sample(range(len(v)), int(0.2 * len(v)))
            subj = (k[0], k[1])
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
        train["label_to_exer_hand"] = unique_indices
        train["info"] = DEFAULT_INFO

        test["data"] = test_data
        test["label"] = test_label
        test["label_to_exer_hand"] = unique_indices
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

        # print(train["data"].keys())

    def concanetate_data(self):
        seed(self.args.dataset_seed)
        data = self.data["data"]
        label = self.data["label"]
        label_to_exer_hand = self.data["label_to_exer_hand"]
        exercise_label = []
        for i in range(len(label_to_exer_hand)):
            if label_to_exer_hand[i][0] not in exercise_label:
                exercise_label.append(label_to_exer_hand[i][0])
        exercise_label = sorted(exercise_label)
        res_data = []
        res_label = []
        res_exer_label = []
        for k, v in data.items():
            file = []
            dense_label = []
            k_label = label[k]
            if self.args.shuffle == "random":
                # bc there isnt a variation we cant shuffle on hand but only on the id:
                combined = list(zip(v, k_label))
            else:  # sorted

                combined = list(zip(v, k_label))  # zip the data and the label
                combined = sort_filename(combined)
            for ind, (filename, original_label) in enumerate(combined):
                df_np = pd.read_csv(filename).to_numpy()[
                    :, 1:7
                ]  # (T, 6), 1 repetition
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
                    [exercise_label.index(label_to_exer_hand[original_label][0])]
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

    def load_data(self, split):
        data = np.load(
            os.path.join(self.root, self.DATASET_NAME, f"{split}_data.npy"),
            allow_pickle=True,
        ).item()
        return data
