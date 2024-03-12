import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from TaskSampler import TaskSampler
import pandas as pd
from utilities import sliding_windows
from utils_dataset import ClassificationDataset, CustomDataset, NormalDataset
from torch.utils.data import Dataset, DataLoader

def get_e2_e4():
    folder = './datasets/physiq/segment_sessions_one_repetition_data_E2'
    files = []
    dict_mapping = dict()
    for file in os.listdir(folder):
        if '.csv' not in file:
            continue
        if '120' not in file:
            continue
        files.append(file)
        subject = file.split('_')[1]
        if subject not in dict_mapping:
            dict_mapping[subject] = [file]
        else:
            dict_mapping[subject].append(file)
            
    inputs = []
    labels = []
    for k, v in dict_mapping.items():
        v = sorted(v)
        v_concat = []
        for i in v:
            path_file = os.path.join(folder, i)
            df = pd.read_csv(path_file)
            v_concat.append(df.iloc[:, 1:7].values)
        e2 = np.concatenate(v_concat, axis=0)
        inputs.append(e2)
        labels.append([0] * e2.shape[0]) # e2
    
    folder = './datasets/physiq/segment_sessions_one_repetition_data_E4'
    files = []
    dict_mapping = dict()
    for file in os.listdir(folder):
        if '.csv' not in file:
            continue
        if '120' not in file:
            continue
        files.append(file)
        subject = file.split('_')[1]
        if subject not in dict_mapping:
            dict_mapping[subject] = [file]
        else:
            dict_mapping[subject].append(file)
    
    for k, v in dict_mapping.items():
        v = sorted(v)
        v_concat = []
        for i in v:
            path_file = os.path.join(folder, i)
            df = pd.read_csv(path_file)
            v_concat.append(df.iloc[:, 1:7].values)
        e4 = np.concatenate(v_concat, axis=0)
        inputs.append(e4)
        labels.append([1] * e4.shape[0])
    inputs = np.concatenate(inputs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return inputs, labels
    


def get_e2_e2prime():
    # e2 is regulr exercise: [0, 200], [200, 400] for example
    # e2 prime is the starting at half and ending at the next half: [100, 300], [300, 0-100]
    folder = './datasets/physiq/segment_sessions_one_repetition_data_E2'
    files = []
    dict_mapping = dict()
    for file in os.listdir(folder):
        if '.csv' not in file:
            continue
        if '120' not in file:
            continue
        files.append(file)
        subject = file.split('_')[1]
        if subject not in dict_mapping:
            dict_mapping[subject] = [file]
        else:
            dict_mapping[subject].append(file)
            
    # permute the order:
    # np.random.shuffle(files)
    inputs = []
    labels = []
        
    # e2 prime:
    for k, v in dict_mapping.items():
        v = sorted(v)
        v_concat = []
        for i in v:
            path_file = os.path.join(folder, i)
            df = pd.read_csv(path_file)
            v_concat.append(df.iloc[:, 1:7].values)
        e2 = np.concatenate(v_concat, axis=0)
        # move v_concat[0][:len(v_concat[0])//2] to the end:
        v_concat.insert(len(v_concat), v_concat[0][:len(v_concat[0])//2])
        v_concat[0] = v_concat[0][len(v_concat[0])//2:]
        e2_prime = np.concatenate(v_concat, axis=0)
        inputs.append(e2_prime)
        labels.append([1] * e2_prime.shape[0]) # e2 prime
        
        inputs.append(e2)
        labels.append([0] * e2.shape[0]) # e2
        
    inputs = np.concatenate(inputs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return inputs, labels

def test_idea_dataloader_e2(config):
    
    inputs, labels = get_e2_e4()
    sw = sliding_windows(50, 15)
    segmented_samples, segmented_labels = sw(torch.tensor(inputs), torch.tensor(labels))
    # Split the dataset into train, val and test:
    train_samples, test_samples, train_labels, test_labels = train_test_split(segmented_samples, segmented_labels, test_size=0.5, random_state=42)
    # val split:
    train_samples, val_samples, train_labels, val_labels = train_test_split(train_samples, train_labels, test_size=0.2, random_state=42)
    train_set = ClassificationDataset(train_samples, train_labels)
    val_set = ClassificationDataset(val_samples, val_labels)
    test_set = ClassificationDataset(test_samples, test_labels)
    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,)
    val_loader = DataLoader(
        val_set,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,)
    test_loader = DataLoader(
        test_set,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,)
    return train_loader, val_loader, test_loader
    
    
        
        

            
            
            
    
    
    


def get_dataloaders(config):
    if config['fsl']:
        return fsl_dataloaders(config)
    else:
        return non_fsl_dataloaders(config)

def non_fsl_dataloaders(config):
    """
    Get dataloaders for the train, val, test sets for non-FSL

    Args:
        config (_type_): NA
    """
    inputs, labels = _load(config)
    sw = sliding_windows(300, 50)
    segmented_samples, segmented_labels = sw(torch.tensor(inputs), torch.tensor(labels))
    # Split the dataset into train, val and test:
    train_samples, test_samples, train_labels, test_labels = train_test_split(segmented_samples, segmented_labels, test_size=0.2, random_state=42)
    train_samples, val_samples, train_labels, val_labels = train_test_split(train_samples, train_labels, test_size=0.2, random_state=42)
    train_set = NormalDataset(train_samples, train_labels)
    val_set = NormalDataset(val_samples, val_labels)
    test_set = NormalDataset(test_samples, test_labels)
    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,)
    val_loader = DataLoader(
        val_set,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,)
    test_loader = DataLoader(
        test_set,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,)
    return train_loader, val_loader, test_loader
        
    

    
    

def fsl_dataloaders(config):
    """
    Get dataloaders for the train and test sets for FSL

    :param config: A dictionary containing the following keys:
        - 'n_way': Number of classes in a task
        - 'n_shot': Number of support examples per class in a task
        - 'n_query': Number of query examples per class in a task
        - 'n_tasks_per_epoch': Number of tasks per epoch
    :return: A tuple of (train_loader, test_loader)
    """
    # Load the dataset
    inputs, labels = _load(config)
    sw = sliding_windows(300, 50)
    segmented_samples, segmented_labels = sw(torch.tensor(inputs), torch.tensor(labels))
    # Split the dataset into train, val and test:
    train_samples, test_samples, train_labels, test_labels = train_test_split(segmented_samples, segmented_labels, test_size=0.5, random_state=42)
    # val split:
    # train_samples, val_samples, train_labels, val_labels = train_test_split(train_samples, train_labels, test_size=0.2, random_state=42)
    train_set = CustomDataset(train_samples, train_labels)
    # val_set = CustomDataset(val_samples, val_labels)
    test_set = CustomDataset(test_samples, test_labels)
    # train_set = CustomDataset(segmented_samples, segmented_labels)
    n_way = config['n_way']
    n_shot = config['n_shot']
    n_query = config['n_query']
    n_tasks_per_epoch = config['n_tasks_per_epoch']
    allowed_label = config['allowed_label']
    batch_size = config['batch_size']
    train_sampler = TaskSampler(train_set, allowed_label, n_way, n_shot, batch_size, n_query, n_tasks_per_epoch)
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )
    # val_loader = DataLoader(
    #     val_set,
    #     batch_sampler=train_sampler,
    #     num_workers=0,
    #     pin_memory=True,
    #     collate_fn=wrapped_collate_fn,
    # )
    test_loader = DataLoader(
        test_set,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )
    return train_loader, test_loader


def _load(config):
    if config['dataset'].lower() == 'physiq_e1':
        return _load_physiq_e1()
    elif config['dataset'].lower() == 'physiq_e2':
        return _load_physiq_e2()
    elif config['dataset'].lower() == 'physiq_e3':
        return _load_physiq_e3()
    elif config['dataset'].lower() == 'opportunity':
        return _load_opportunity()
    else:
        raise ValueError(f'Unknown dataset: {config["dataset"]}')
    
def _load_physiq_e1():
    inp = np.load('./datasets/physiq/physiq_permute_e1.npy', allow_pickle=True)
    inputs, labels = inp.item()['inputs'], inp.item()['labels']
    return inputs, labels

def _load_physiq_e2():
    inp = np.load('./datasets/physiq/physiq_permute_e2.npy', allow_pickle=True)
    inputs, labels = inp.item()['inputs'], inp.item()['labels']
    return inputs, labels

def _load_physiq_e3():
    inp = np.load('./datasets/physiq/physiq_permute_e3.npy', allow_pickle=True)
    inputs, labels = inp.item()['inputs'], inp.item()['labels']
    return inputs, labels


def _load_opportunity():
    inp = np.load('./datasets/OpportunityUCIDataset/loco_2_mask.npy', allow_pickle=True)
    inputs, labels = inp.item()['inputs'], inp.item()['labels']
    return inputs, labels
    
class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)