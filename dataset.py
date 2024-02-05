from functools import partial
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from utilities import * 


def default_splitter(folder_name, config: dict, split=False, transform=None):
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
    method = config['preprocess']['method']

    width = config['preprocess']['width']
    step = config['preprocess']['step']

    file_names = []
    for file in os.listdir(folder_name):
        if os.path.isfile(os.path.join(folder_name, file)) and file.endswith('.csv'):
            file_names.append(os.path.join(folder_name, file))
        else:
            print(f'{file} is not a csv file')
    # permute the order:
    file_names = np.random.permutation(file_names)
    
    # partial insertion of variables:
    if method == 'default':
        dataset = partial(DefaultDataset, width=width, step=step, transform=transform)
    elif method == 'noise-both-end':
        dataset = partial(NoiseDataset, width=width, step=step, transform=transform)
    elif method == 'classification':
        classes = []
        # re-organize the class label as it does not follow the convention: 0 to N-1
        for file in file_names:
            class_label = int(file.split('/')[-1].split('_')[1][1]) #NOTE: assuming '1' is the class label
            if class_label not in classes:
                classes.append(class_label)
        classes = sorted(classes)
        # print in red:
        printc(f'Total of Classes: {classes}', 'red')
        dataset = partial(ClassificationDataset, classes=classes, width=width, step=step, transform=transform)
    else:
        raise NotImplementedError

    if split:
        train_file_names = file_names[:int(len(file_names)*0.8)]
        val_file_names = file_names[int(len(file_names)*0.8):int(len(file_names)*0.9)]
        test_file_names = file_names[int(len(file_names)*0.9):]
        return dataset(train_file_names), dataset(val_file_names), dataset(test_file_names)
    else:
        # 2 ways split
        train_file_names = file_names[:int(len(file_names)*0.8)]
        test_file_names = file_names[int(len(file_names)*0.8):]
        return dataset(train_file_names), dataset(test_file_names)

class DefaultDataset(Dataset):
    """
    This is a custom dataset class for time series data.
    Specifically, it is using the typical way to load all the data into memory and 
    return data and label in __getitem__ method.

    TODO: add standardscaler into it to scale the data
    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, file_names, scaler=None, width=50, step=25, transform=None):
        self.file_names = file_names
        self.width = width
        self.step = step
        self.x, self.y = self._load_all_data()
        self.transform = transform
        
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.x[idx]), self.y[idx]
        return self.x[idx], self.y[idx]
    
    def _load_all_data(self):
        # load into memory and do sliding windows:
        sw = sliding_windows(self.width, self.step)
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
            

class NoiseDataset(Dataset):
    """  
    This is a custom dataset class for time series data.
    Specifically, it is using the typical way to load all the data into memory and 
    return data and label in __getitem__ method. 
    The difference is that it adds noise to the data.
    """
    def __init__(self, file_names, scaler=None, width=50, step=25, transform=None):
        self.file_names = file_names
        self.width = width
        self.step = step
        self.x, self.y = self._load_all_data()
        self.transform = transform
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.x[idx]), self.y[idx]
        return self.x[idx], self.y[idx]
    def _add_noise(self, data):
        # add noise on both end [50-100] + [data] + [50-100]
        # length is between 50 and 100 randomly selected:
        begin_length = np.random.randint(50, 100)
        begin_noise = torch.zeros((begin_length, data.shape[1]))
        # the noise is simliar to the data[0] or data[-1] based on the side:
        begin_noise += data[0, :]
        # add some noise to it:
        begin_noise = begin_noise + torch.randn(begin_noise.shape) * 0.1
        
        end_length = np.random.randint(50, 100)
        end_noise = torch.zeros((end_length, data.shape[1]))
        end_noise += data[-1, :]
        end_noise = end_noise + torch.randn(end_noise.shape) * 0.1
        # concatenate:
        data = torch.cat([begin_noise, data, end_noise])
        return data, (begin_length, end_length)
            


    def _load_all_data(self):
        # load into memory and do sliding windows:
        sw = sliding_windows(self.width, self.step)
        # load data into numpy:
        x = []
        y = []
        for file in self.file_names:
            data = pd.read_csv(file)
            data = data.values
            data = torch.from_numpy(data)[:, 1::].float()
            # add noise:
            data, (begin_length, end_length) = self._add_noise(data)
            # create label length_0 = 0, length_-1 = 0, length_1:n-1 = 1
            label = torch.zeros(data.shape[0])
            label[begin_length:-end_length] = 1
            # do sliding windows:
            data, label = sw(data, label)
            # append to the list
            x.append(data)
            y.append(label)
        return torch.cat(x), torch.cat(y)



class ClassificationDataset:
    def __init__(self, file_names, classes, width=50, step=25, transform=None,):
        self.file_names = file_names
        self.width = width
        self.step = step
        self.transform = transform
        self.classes = classes
        self.x, self.y = self._load_all_data()
        
        
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.x[idx]), self.y[idx]
        return self.x[idx], self.y[idx]
    
    def _load_all_data(self):
        # load into memory and do sliding windows:
        sw = sliding_windows(self.width, self.step)
        # load data into numpy:
        x = []
        y = []
        for file in self.file_names:
            class_label = int(file.split('/')[-1].split('_')[1][1])
            data = pd.read_csv(file)
            data = data.values
            data = torch.from_numpy(data)[:, 1::].float()
            if data.shape[0] <= self.width:
                data = self._interpolate(data) #NOTE: interpolate the data to the same length otherwise it will be a problem
            placeholder_label = torch.zeros(data.shape[0]) #NOTE: placeholder in case of classification
            # do sliding windows:
            data, _ = sw(data, placeholder_label)
            # append to the list
            x.append(data)
            # y.append(label)
            # use its index as the label:
            class_label_idx = self.classes.index(class_label)
            y.extend([class_label_idx] * data.shape[0])
        return torch.cat(x), torch.tensor(y) # torch.cat(y)
    
    def _interpolate(self, data):
        from scipy import interpolate
        # interpolate the data to the same length:
        # randomly select a length:
        length = self.width
        # interpolate using nummpy:
        original_indices = np.arange(data.shape[0])
        new_indices = np.linspace(0, data.shape[0]-1 , length)

        # Interpolated array
        interpolated_array = np.zeros((length, 6))

        for i in range(6):
            interp_func = interpolate.interp1d(original_indices, data[:, i])
            interpolated_array[:, i] = interp_func(new_indices)
        return torch.tensor(interpolated_array).float()
if __name__ == '__main__':
    # Usage example
    folder_name = './datasets/spar9x'
    config = {'preprocess': {'method': 'classification', 'width': 50, 'step': 25}, 'model': 'lstm'}
    train_dataset, test_dataset = default_splitter(folder_name, config, split=False)
    len(train_dataset)
