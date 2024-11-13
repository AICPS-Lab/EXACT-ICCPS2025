from sklearn.model_selection import train_test_split
import torch

from ..datasets.DenseLabelTaskSampler import DenseLabelTaskSampler
from utilities import sliding_windows
from archives.utils_dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader


def fsl_dataloaders(
    window_size,
    window_step,
    n_way,
    n_shot,
    n_query,
    n_tasks_per_epoch,
    batch_size,
):
    """
    Get dataloaders for the train and test sets for FSL

    :param config: A dictionary containing the following keys:
        - 'n_way': Number of classes in a task
        - 'n_shot': Number of support examples per class in a task
        - 'n_query': Number of query examples per class in a task
        - 'n_tasks_per_epoch': Number of tasks per epoch
    :return: A tuple of (train_loader, test_loader)
    """
    
    # train_set = CustomDataset(segmented_samples, segmented_labels)
    train_sampler = DenseLabelTaskSampler(
        train_set, n_way, n_shot, batch_size, n_query, n_tasks_per_epoch
    )
    test_sampler = DenseLabelTaskSampler(
        test_set, n_way, n_shot, batch_size, n_query, n_tasks_per_epoch
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_sampler=test_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=test_sampler.episodic_collate_fn,
    )
    return train_loader, test_loader
