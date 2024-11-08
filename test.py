from matplotlib import pyplot as plt
from datasets.MMFIT import MMFIT
from datasets.SPAR import SPAR
from datasets.DenseLabelTaskSampler import DenseLabelTaskSampler
from datasets.PhysiQ import PhysiQ
from torch.utils.data import Dataset, DataLoader
import wandb
import random

from datasets.Transforms import IMUAugmentation
from until_argparser import get_args
from utilities import model_exception_handler

# start a new wandb run to track this script
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",

#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.02,
#     "architecture": "CNN",
#     "dataset": "CIFAR-100",
#     "epochs": 10,
#     }
# )

# # simulate training
# epochs = 10
# offset = random.random() / 5
# for epoch in range(2, epochs):
#     acc = 1 - 2 ** -epoch - random.random() / epoch - offset
#     loss = 2 ** -epoch + random.random() / epoch + offset

#     # log metrics to wandb
#     wandb.log({"acc": acc, "loss": loss})

# # [optional] finish the wandb run, necessary in notebooks
# wandb.finish()

def test_num_workers():
    from argparse import Namespace
    from time import time
    import multiprocessing as mp
    from torch.utils.data import Dataset, DataLoader
    from datasets.DenseLabelTaskSampler import DenseLabelTaskSampler
    from datasets.PhysiQ import PhysiQ
    from datasets.Transforms import IMUAugmentation
    from until_argparser import get_args

    args = get_args()  # Get arguments from the argparse

    print(args.loocv)
    dataset = SPAR(
        root="data",
        split="train",
        window_size=args.window_size,
        bg_fg=None,
        args=args,
        transforms=IMUAugmentation(rotation_chance=args.rotation_chance),
    )

    train_sampler = DenseLabelTaskSampler(
        dataset,
        # n_way=1,
        n_shot=1,
        batch_size=64,
        n_query=1,
        n_tasks=5,
        threshold_ratio=0.25,
        add_side_noise=args.add_side_noise,
    )

    for num_workers in range(0, mp.cpu_count(), 2):  
        train_loader = DataLoader(
            dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=train_sampler.episodic_collate_fn,
        )
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

def test_setup():
    args = get_args()  # Get arguments from the argparse
    args.add_side_noise = True
    print(args.loocv)
    dataset = SPAR(
        root="data",
        split="train",
        window_size=args.window_size,
        window_step=args.window_step,
        bg_fg=None,
        args=args,
        transforms=IMUAugmentation(rotation_chance=args.rotation_chance),
    )
    
    # dataset = SPAR(
    #     root="data",
    #     split="train",
    #     window_size=1000,
    #     bg_fg=None,
    #     args=args,
    #     transforms=IMUAugmentation(rotation_chance=0),
    # )

    # for i in range(len(dataset)):
    #     print(dataset[i][1].unique())

    train_sampler = DenseLabelTaskSampler(
        dataset,
        # n_way=1,
        n_shot=1,
        batch_size=args.batch_size,
        n_query=1,
        n_tasks=5,
        threshold_ratio=0.25,
        add_side_noise=args.add_side_noise,
    )
    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )

    for (
        support_images,
        support_labels,
        query_images,
        query_labels,
        true_class_ids,
    ) in train_loader:
        print(
            support_images.shape,
            support_labels.shape,
            query_images.shape,
            query_labels.shape,
            true_class_ids,
        )
        # print(true_class_ids)
        # for i in range(8):
        #subplot of support:
        ax = plt.subplot(121)
        plt.plot(support_images[0, 0])
        plt.plot(support_labels[0, 0])
        #subplot of query:
        ax = plt.subplot(122)
        plt.plot(query_images[0, 0])
        plt.plot(query_labels[0, 0])
        plt.show()

if __name__ == "__main__":
    # test the dataset
    # seed(42)
    
    # test_num_workers()
    
    test_setup()
