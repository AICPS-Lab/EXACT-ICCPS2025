import random
import torch
from torch import Tensor
from torch.utils.data import Sampler, Dataset
from typing import Dict, Iterator, List, Tuple, Union

class TaskSampler(Sampler):
    """ Samples batches in the shape of few-shot segmentation tasks. 
    At each iteration, it will sample n_way classes, and then sample support and query multivariate time series (mts) from these classes.
    #TODO: unfinished
    """
    def __init__(self, dataset: Dataset, n_way: int, n_shot: int, batch_size: int, n_query: int, n_tasks: int):
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.batch_size = batch_size
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.dataset = dataset
        self.items_per_label: Dict[int, List[int]] = {}
        for item, label in enumerate(dataset.get_labels()):
            if label in self.items_per_label:
                self.items_per_label[label].append(item)
            else:
                self.items_per_label[label] = [item]
        self._check_dataset_size_fits_sampler_parameters()
        
    def __len__(self) -> int:
        return self.n_tasks
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Sample n_way labels uniformly at random,
        and then sample n_shot + n_query items for each label, also uniformly at random.
        Yields:
            a list of indices of length (n_way * batch_size * (n_shot + n_query))
        """
        for _ in range(self.n_tasks):
            for label in random.sample(sorted(self.items_per_label.keys()), self.n_way):
                yield torch.cat(
                    [
                        torch.tensor(
                            random.sample(
                                self.items_per_label[label], (self.n_shot + self.n_query) * self.batch_size
                            )
                        )
                    ]
                )
            
        
    
    def _check_dataset_size_fits_sampler_parameters(self):
        """
        Check that the dataset size is compatible with the sampler parameters
        """
        self._check_dataset_has_enough_labels()
        self._check_dataset_has_enough_items_per_label()

    def _check_dataset_has_enough_labels(self):
        if self.n_way > len(self.items_per_label):
            raise ValueError(
                f"The number of labels in the dataset ({len(self.items_per_label)} "
                f"must be greater or equal to n_way ({self.n_way})."
            )

    def _check_dataset_has_enough_items_per_label(self):
        number_of_samples_per_label = [
            len(items_for_label) for items_for_label in self.items_per_label.values()
        ]
        minimum_number_of_samples_per_label = min(number_of_samples_per_label)
        label_with_minimum_number_of_samples = number_of_samples_per_label.index(
            minimum_number_of_samples_per_label
        )
        if self.n_shot + self.n_query > minimum_number_of_samples_per_label:
            raise ValueError(
                f"Label {label_with_minimum_number_of_samples} has only {minimum_number_of_samples_per_label} samples"
                f"but all classes must have at least n_shot + n_query ({self.n_shot + self.n_query}) samples."
            )