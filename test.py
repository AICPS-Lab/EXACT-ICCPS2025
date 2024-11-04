from matplotlib import pyplot as plt
from datasets.DenseLabelTaskSampler import DenseLabelTaskSampler
from datasets.PhysiQ import PhysiQ
from torch.utils.data import Dataset, DataLoader
import wandb
import random

from datasets.Transforms import IMUAugmentation
from until_argparser import get_args

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
if __name__ == "__main__":
    # test the dataset

    args = get_args()  # Get arguments from the argparse
    dataset = PhysiQ(
        root="data",
        split="train",
        window_size=1000,
        bg_fg=None,
        args=args,
        transforms=IMUAugmentation(rotation_chance=0),
    )

    # for i in range(len(dataset)):
    #     print(dataset[i][1].unique())

    train_sampler = DenseLabelTaskSampler(
        dataset,
        # n_way=1,
        n_shot=1,
        batch_size=64,
        n_query=1,
        n_tasks=5,
        threshold_ratio=0.25,
    )
    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )
    # support_images, support_labels, query_images, query_labels, true_class_ids = next(iter(train_loader))
    # print(
    #     support_images.shape,
    #     support_labels.shape,
    #     query_images.shape,
    #     query_labels.shape,
    #     true_class_ids,
    # )
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
        plt.plot(support_images[0, 0])
        plt.plot(support_labels[0, 0])
        plt.show()
