import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

class sliding_windows(torch.nn.Module):
    def __init__(self, total_length, width, step):
        # https://stackoverflow.com/questions/53972159/how-does-pytorchs-fold-and-unfold-work
        super(sliding_windows, self).__init__()
        self.total_length = total_length
        self.width = width
        self.step = step

    def forward(self, input_time_series, labels):
        input_transformed = torch.swapaxes(input_time_series.unfold(-2, size=self.width, step=self.step), -2, -1)
        # For labels, we only have one dimension, so we unfold along that dimension
        labels_transformed = labels.unfold(0, self.width, self.step)
        return input_transformed, labels_transformed

    def get_num_sliding_windows(self):
        return round((self.total_length - (self.width - self.step)) / self.step)

def default_splitter(folder_name, split=False):
    """This is a custom dataset class for time series data.
    Specifically, it is using the typical way to load all the data into memory and 
    return data and label in __getitem__ method.

    Args:
        file_name (_type_): _description_
        split (bool, optional): if split False, only two ways, 
                                otherwise 3 ways. Defaults to False.

    Returns:
        [Datasets]: _description_
    """
    file_names = []
    for file in os.listdir(folder_name):
        if os.path.isfile(os.path.join(folder_name, file)) and file.endswith('.csv'):
            file_names.append(os.path.join(folder_name, file))
        else:
            print(f'{file} is not a csv file')
    # permute the order:
    file_names = np.random.permutation(file_names)
    
    if split:
        train_file_names = file_names[:int(len(file_names)*0.8)]
        val_file_names = file_names[int(len(file_names)*0.8):int(len(file_names)*0.9)]
        test_file_names = file_names[int(len(file_names)*0.9):]
        return DefaultDataset(train_file_names), DefaultDataset(val_file_names), DefaultDataset(test_file_names)
    else:
        # 2 ways split
        train_file_names = file_names[:int(len(file_names)*0.8)]
        test_file_names = file_names[int(len(file_names)*0.8):]
        return DefaultDataset(train_file_names), DefaultDataset(test_file_names)

class DefaultDataset(Dataset):
    """
    This is a custom dataset class for time series data.
    Specifically, it is using the typical way to load all the data into memory and 
    return data and label in __getitem__ method.

    TODO: add standardscaler into it to scale the data
    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, file_names):
        self.file_names = file_names
        self.x, self.y = self._load_all_data()
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def _load_all_data(self):
        # load into memory and do sliding windows:
        sw = sliding_windows(50, 25, 20)
        # load data into numpy:
        x = []
        y = []
        for file in self.file_names:
            data = pd.read_csv(file)
            data = data.values
            data = torch.from_numpy(data)[:, 1::].float()
            # create label length_0 = 0, length_-1 = 0, length_1:n-1 = 1
            label = torch.zeros(data.shape[0])
            label[1:-1] = 1
            # do sliding windows:
            data, label = sw(data, label)
            # append to the list
            x.append(data)
            y.append(label)
        return torch.cat(x), torch.cat(y)
            
        
    
if __name__ == '__main__':
    # Usage example
    folder_name = './datasets/spar'
    train_dataset, test_dataset = default_splitter(folder_name, split=False)
    len(train_dataset)
