import numpy as np
from sklearn.model_selection import train_test_split
import torch
from task_sampler import TaskSampler

from utilities import sliding_windows
from utils_dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader

def get_dataloaders(config):
    """
    Get dataloaders for the train and test sets.

    :param config: A dictionary containing the following keys:
        - 'n_way': Number of classes in a task
        - 'n_shot': Number of support examples per class in a task
        - 'n_query': Number of query examples per class in a task
        - 'n_tasks_per_epoch': Number of tasks per epoch
    :return: A tuple of (train_loader, test_loader)
    """
    # Load the dataset
    inp = np.load('./datasets/OpportunityUCIDataset/loco_2_mask.npy', allow_pickle=True)
    inputs, labels = inp.item()['inputs'], inp.item()['labels']
    sw = sliding_windows(300, 50)
    segmented_samples, segmented_labels = sw(torch.tensor(inputs), torch.tensor(labels))
    # Split the dataset into train, val and test:
    train_samples, test_samples, train_labels, test_labels = train_test_split(segmented_samples, segmented_labels, test_size=0.2, random_state=42)
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
    train_sampler = TaskSampler(train_set, n_way, n_shot, n_query, n_tasks_per_epoch)
    def wrapped_collate_fn(batch):
        # 5 -> 7 of tuples:
        # (support_set, query_set, support_labels, query_labels, classes)
        original_output = train_sampler.episodic_collate_fn(batch)
        (example_support_images, example_support_labels, example_query_images, example_query_labels, example_class_ids )= original_output
        example_support_images, example_support_images_labels = example_support_images[:, :, :-1], example_support_images[:, :, -1]
        example_query_images, example_query_images_labels = example_query_images[:, :, :-1], example_query_images[:, :, -1]
        example_support_images = [ [example_support_images[i+j, :, :].transpose(0, 1).unsqueeze(0) for j in range(n_shot)] for i in range(n_way)]
        example_support_images_labels = [ [example_support_images_labels[i+j, :].unsqueeze(0) for j in range(n_shot)] for i in range(n_way)]
        example_support_labels = [example_support_labels[i].unsqueeze(0) for i in range(n_way)]
        example_query_images = [example_query_images[i, :, :].transpose(0, 1).unsqueeze(0) for i in range(n_query)]
        example_query_images_labels = [example_query_images_labels[i, :] for i in range(n_query)]
        example_query_labels = [example_query_labels[i] for i in range(n_query)]
        return (example_support_images, example_support_images_labels, example_support_labels, example_query_images, example_query_images_labels, example_query_labels, example_class_ids)
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=wrapped_collate_fn,
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
        collate_fn=wrapped_collate_fn,
    )
    return train_loader, test_loader


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