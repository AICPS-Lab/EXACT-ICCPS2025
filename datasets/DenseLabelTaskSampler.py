import random
import torch
from torch import Tensor
from torch.utils.data import Sampler, Dataset
from typing import Dict, Iterator, List, Tuple, Union


class DenseLabelTaskSampler(Sampler):
    """
    Samples batches for few-shot tasks in a dense labeling setup with threshold-based label classification.
    """

    def __init__(
        self,
        dataset: Dataset,
        n_way: int,
        n_shot: int,
        batch_size: int,
        n_query: int,
        n_tasks: int,
        threshold_ratio: float,
    ):
        """
        Args:
            dataset: The dataset from which to sample.
            n_way: Number of different classes per task (i.e., the number of unique classes to sample in each task).
            n_shot: Number of examples per class in the support set.
            batch_size: Number of batches.
            n_query: Number of examples per class in the query set.
            n_tasks: Number of tasks to sample.
            threshold_ratio: The minimum ratio of non-background labels required to consider a label valid (e.g., 50/300).
        """
        super().__init__(data_source=None)
        self.n_way = n_way
        self.n_shot = n_shot
        self.batch_size = batch_size
        self.n_query = n_query
        self.n_tasks = n_tasks
        self.threshold_ratio = (
            threshold_ratio  # Ratio for non-background labels
        )

        self.dataset = dataset
        self.items_per_label: Dict[int, List[int]] = {}

        # Build a dictionary mapping each label to a list of indices for dense labeling
        for item_idx, (input_data, label) in enumerate(dataset):
            valid_label = self._classify_label(label)
            if valid_label is not None:
                if valid_label in self.items_per_label:
                    self.items_per_label[valid_label].append(item_idx)
                else:
                    self.items_per_label[valid_label] = [item_idx]

        self._check_dataset_size_fits_sampler_parameters()

    def __len__(self) -> int:
        return self.n_tasks

    def __iter__(self) -> Iterator[List[int]]:
        """
        Sample `n_way` labels, and for each label, sample `n_shot` + `n_query` items.
        Only labels that meet the threshold condition will be sampled.
        Yields:
            A list of indices of length (n_way * batch_size * (n_shot + n_query)).
        """
        for _ in range(self.n_tasks):
            sampled_task_indices = torch.cat(
                [
                    torch.tensor(
                        random.sample(
                            self.items_per_label[label],
                            (self.n_shot + self.n_query) * self.batch_size,
                        )
                    )
                    for label in random.sample(
                        sorted(self.items_per_label.keys()), self.n_way
                    )
                ]
            )
            yield sampled_task_indices.tolist()

    def episodic_collate_fn(
        self, input_data: List[Tuple[Tensor, Tensor]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]:
        """
        Collate function for episodic data loaders in dense labeling problems.
        Args:
            input_data: List of tuples where each element contains:
                - An input sample as a torch Tensor (e.g., image or time series).
                - A corresponding dense label as a torch Tensor (e.g., segmentation mask).
        Returns:
            Tuple containing:
                - support_images: Support set inputs of shape (n_way * n_shot * batch_size, channels, height, width).
                - support_labels: Support set dense labels of the same shape as the input (n_way * n_shot * batch_size, height, width).
                - query_images: Query set inputs of shape (n_way * n_query * batch_size, channels, height, width).
                - query_labels: Query set dense labels of the same shape as the input (n_way * n_query * batch_size, height, width).
                - true_class_ids: The true class IDs of the classes sampled in the episode.
        """
        true_class_ids = list(
            {self._classify_label(x[1]) for x in input_data}
        )  # Use max value to identify class IDs in dense labels
        all_images = torch.cat(
            [x[0].unsqueeze(0) for x in input_data]
        )  # Stack all input samples
        all_labels = torch.cat(
            [x[1].unsqueeze(0) for x in input_data]
        )  # Stack all dense labels

        all_images = all_images.reshape(
            (
                self.n_way,
                self.n_shot + self.n_query,
                self.batch_size,
                *all_images.shape[1:],
            )
        )
        all_labels = all_labels.reshape(
            (
                self.n_way,
                self.n_shot + self.n_query,
                self.batch_size,
                *all_labels.shape[1:],
            )
        )

        # Separate support and query sets
        support_images = all_images[:, : self.n_shot].reshape(
            (-1, *all_images.shape[3:])
        )
        query_images = all_images[:, self.n_shot :].reshape(
            (-1, *all_images.shape[3:])
        )

        support_labels = all_labels[:, : self.n_shot].reshape(
            (-1, *all_labels.shape[3:])
        )
        query_labels = all_labels[:, self.n_shot :].reshape(
            (-1, *all_labels.shape[3:])
        )

        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )

    def _classify_label(self, label: Tensor) -> Union[int, None]:
        """
        Classify the label as 0 or 1 based on the threshold ratio of non-background (non-0) elements.
        Args:
            label: A tensor representing the dense label for a sample.
        Returns:
            - 1 if the label contains a sufficient percentage of non-background elements (above threshold).
            - 0 if the label is entirely background (below threshold).
            - None if the label is ignored due to being between 0 and the threshold.
        """
        total_elements = label.numel()
        non_bg_elements = (label > 0).sum().item()

        if non_bg_elements == 0:
            return 0  # Entirely background
        elif non_bg_elements / total_elements >= self.threshold_ratio:
            return 1  # Sufficient non-background elements
        else:
            return None  # Ignore this label

    def _check_dataset_size_fits_sampler_parameters(self):
        """
        Check that the dataset size is compatible with the sampler parameters.
        """
        self._check_dataset_has_enough_labels()
        self._check_dataset_has_enough_items_per_label()

    def _check_dataset_has_enough_labels(self):
        if self.n_way > len(self.items_per_label):
            raise ValueError(
                f"The number of labels in the dataset ({len(self.items_per_label)}) must be greater or equal to n_way ({self.n_way})."
            )

    def _check_dataset_has_enough_items_per_label(self):
        number_of_samples_per_label = [
            len(items_for_label)
            for items_for_label in self.items_per_label.values()
        ]
        minimum_number_of_samples_per_label = min(number_of_samples_per_label)
        label_with_minimum_number_of_samples = (
            number_of_samples_per_label.index(
                minimum_number_of_samples_per_label
            )
        )
        if (
            self.n_shot + self.n_query
        ) * self.batch_size > minimum_number_of_samples_per_label:
            raise ValueError(
                f"Label {label_with_minimum_number_of_samples} has only {minimum_number_of_samples_per_label} samples, "
                f"but all classes must have at least n_shot + n_query ({self.n_shot + self.n_query}) samples."
            )
