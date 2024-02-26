import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd

def majority_vote(series):
    """
    Convert a single time series of shape (300,) to its majority-vote class.

    :param series: np.array of shape (300,), where each element is a class label.
    :return: The majority class for the time series.
    """
    counts = np.bincount(series)
    return np.argmax(counts)


class CustomDataset(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform
        self.class_labels = [majority_vote(self.label[idx]) for idx in range(len(self.label))]
        assert len(self.data) == len(self.label)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if self.transform:
            raise NotImplementedError
            return self.transform(self.data[idx]), self.label[idx]
        cur_label = self.label[idx]
        cur_label = np.where(cur_label == 0, 0, 1)
        return torch.tensor(np.concatenate((self.data[idx], cur_label[np.newaxis].T), axis=1)), torch.tensor(self.class_labels[idx], dtype=torch.int16)
    def get_labels(self):
        return self.class_labels