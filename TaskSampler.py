import random
import torch
from torch import Tensor
from torch.utils.data import Sampler, Dataset
from typing import Dict, Iterator, List, Tuple, Union
GENERIC_TYPING_ERROR_MESSAGE = (
    "Check out the output's type of your dataset's __getitem__() method."
    "It must be a Tuple[Tensor, int] or Tuple[Tensor, 0-dim Tensor]."
)
# refernece: https://github.com/sicara/easy-few-shot-learning/blob/2ed387ef1e30176c679657ccca1b1a2b0ca247a4/easyfsl/samplers/task_sampler.py#L16
class TaskSampler(Sampler):
    """ Samples batches in the shape of few-shot segmentation tasks. 
    At each iteration, it will sample n_way classes, and then sample support and query multivariate time series (mts) from these classes.
    #TODO: unfinished
    """
    def __init__(self, dataset: Dataset, fg_label: List[int], n_way: int, n_shot: int, batch_size: int, n_query: int, n_tasks: int):
        """
        In tasksampler have a param (fg_label); e.g., class = [0,1,2,3,4], fg_label = [1,2], therefore, n_ways can only be ≤ 2 bc len(fg_label), 
        and the rest of [0,3,4] will be as bg_label by default and should always be “0” in segmentation_masks. 
        """
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
        
        self.fg_label = fg_label
        assert fg_label != [], "fg_label: the length of fg_label should be greater than 0"
        assert fg_label != [0], "fg_label: the value of fg_label should be greater than 0, assuming that 0 is the background label"
        assert all([label in self.items_per_label.keys() for label in fg_label]), f"fg_label: {fg_label} should be in the dataset labels: {self.items_per_label.keys()}"
        self.bg_label = [label for label in self.items_per_label.keys() if label not in fg_label]
        assert len(fg_label) <= n_way, f"fg_label: the length of {fg_label} should be less than or equal to n_way: {n_way}"


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
            for label in random.sample(self.fg_label, self.n_way):
                yield torch.cat(
                    [
                        torch.tensor(
                            random.sample(
                                self.items_per_label[label], (self.n_shot + self.n_query) * self.batch_size
                            )
                        )
                    ]
                )
    def episodic_collate_fn(
        self, input_data: List[Tuple[Tensor, Union[Tensor, int]]]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[int]]:
        """
        Collate function to be used as argument for the collate_fn parameter of episodic
            data loaders.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor of shape (n_channels, height, width)
                - the label of this image as an int or a 0-dim tensor
        Returns:
            tuple(Tensor, Tensor, Tensor, Tensor, list[int]): respectively:
                - support images of shape (n_way * n_shot * batch_size, n_channels, height, width),
                - their labels of shape (n_way * n_shot * batch_size),
                - query images of shape (n_way * n_query * batch_size, n_channels, height, width)
                - their labels of shape (n_way * n_query * batch_size),
                - the dataset class ids of the class sampled in the episode
        """
        input_data_with_int_labels = self._cast_input_data_to_tensor_int_tuple(
            input_data
        )
        true_class_ids = list({x[1] for x in input_data_with_int_labels})
        all_images = torch.cat([x[0].unsqueeze(0) for x in input_data_with_int_labels])
        all_images = all_images.reshape(
            (self.n_way, self.n_shot + self.n_query, *all_images.shape[1:])
        )
        all_labels = torch.tensor(
            [true_class_ids.index(x[1]) for x in input_data_with_int_labels]
        ).reshape((self.n_way, self.n_shot + self.n_query))
        support_images = all_images[:, : self.n_shot].reshape(
            (-1, *all_images.shape[2:])
        )
        query_images = all_images[:, self.n_shot :].reshape((-1, *all_images.shape[2:]))
        support_labels = all_labels[:, : self.n_shot].flatten()
        query_labels = all_labels[:, self.n_shot :].flatten()
        return (
            support_images,
            support_labels,
            query_images,
            query_labels,
            true_class_ids,
        )

    def _cast_input_data_to_tensor_int_tuple(
        input_data: List[Tuple[Tensor, Union[Tensor, int]]]
    ) -> List[Tuple[Tensor, int]]:
        """
        Check the type of the input for the episodic_collate_fn method, and cast it to the right type if possible.
        Args:
            input_data: each element is a tuple containing:
                - an image as a torch Tensor of shape (n_channels, height, width)
                - the label of this image as an int or a 0-dim tensor
        Returns:
            the input data with the labels cast to int
        Raises:
            TypeError : Wrong type of input images or labels
            ValueError: Input label is not a 0-dim tensor
        """
        for image, label in input_data:
            if not isinstance(image, Tensor):
                raise TypeError(
                    f"Illegal type of input instance: {type(image)}. "
                    + GENERIC_TYPING_ERROR_MESSAGE
                )
            if not isinstance(label, int):
                if not isinstance(label, Tensor):
                    raise TypeError(
                        f"Illegal type of input label: {type(label)}. "
                        + GENERIC_TYPING_ERROR_MESSAGE
                    )
                if label.dtype not in {
                    torch.uint8,
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                }:
                    raise TypeError(
                        f"Illegal dtype of input label tensor: {label.dtype}. "
                        + GENERIC_TYPING_ERROR_MESSAGE
                    )
                if label.ndim != 0:
                    raise ValueError(
                        f"Illegal shape for input label tensor: {label.shape}. "
                        + GENERIC_TYPING_ERROR_MESSAGE
                    )

        return [(image, int(label)) for (image, label) in input_data]

    
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