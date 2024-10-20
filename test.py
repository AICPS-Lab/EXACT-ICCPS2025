from matplotlib import pyplot as plt
from datasets.DenseLabelTaskSampler import DenseLabelTaskSampler
from datasets.PhysiQ import PhysiQ
from torch.utils.data import Dataset, DataLoader
import wandb
import random

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

    dataset = PhysiQ(
        root="data", N_way=2, split="train", window_size=200, bg_fg=None
    )
    train_sampler = DenseLabelTaskSampler(
        dataset,
        n_way=2,
        n_shot=4,
        batch_size=4,
        n_query=4,
        n_tasks=dataset.NUM_TASKS,
        threshold_ratio=0.25,
    )
    train_loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=train_sampler.episodic_collate_fn,
    )
    
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
    )  in train_loader:
        # print(true_class_ids)
        # for i in range(8):
        plt.plot(support_images[0,0])
        plt.plot(support_labels[0,0])
        plt.show()
