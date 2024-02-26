from few_shot_models import time_FewShotSeg
from methods import Segmenter
from matplotlib import pyplot as plt
from utilities import get_device, sliding_windows
import os
import numpy as np
import pandas as pd
import torch
# custom Dataset:
from torch.utils.data import Dataset, DataLoader
from easyfsl.samplers import TaskSampler
from sklearn.model_selection import train_test_split
from utils_dataset import CustomDataset
    
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
        example_support_images = [ [example_support_images[i+j, :, :].unsqueeze(0) for j in range(n_shot)] for i in range(n_way)]
        example_support_images_labels = [ [example_support_images_labels[i+j, :].unsqueeze(0) for j in range(n_shot)] for i in range(n_way)]
        example_support_labels = [example_support_labels[i].unsqueeze(0) for i in range(n_way)]
        example_query_images = [example_query_images[i, :, :].unsqueeze(0) for i in range(n_query)]
        example_query_images_labels = [example_query_images_labels[i, :].unsqueeze(0) for i in range(n_query)]
        example_query_labels = [example_query_labels[i].unsqueeze(0) for i in range(n_query)]
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

def main():
    config = {
        'n_way': 5,
        'n_shot': 4,
        'n_query': 2,
        'n_tasks_per_epoch': 500,
        'align': True,
    }
    train_loader, test_loader = get_dataloaders(config)
    device = torch.device('cpu')
    segmenter = Segmenter(embed_dims=64, num_classes=2).float().to(device)
    model = time_FewShotSeg(segmenter, device=device, cfg=config).float().to(device)
    
    # train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    for i_iter, sample_batch in enumerate(train_loader):
        
        support_images, support_mask, support_labels, query_images, query_mask, query_labels, classes = sample_batch
        fg_mask = support_mask
        bg_mask = [[torch.where(support_mask[i][j] ==1, 0, 1) for j in range(len(support_mask[1]))] for i in range(len(support_mask))]
        model(support_images, fg_mask, bg_mask, query_images)
       
    
    
    
    
if __name__ == "__main__":
    main()
    
        