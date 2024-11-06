import argparse
import os
import time
import typing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import higher
import wandb

from datasets.DenseLabelTaskSampler import DenseLabelTaskSampler
from torch.utils.data import DataLoader

from until_argparser import get_args, get_dataset, get_model
from loss_fn import (
    dice_coefficient_time_series,
    euclidean_distance_time_series,
    iou_time_series,
)
from methods import EX, UNet
from utils_metrics import visualize_softmax


def train(db, net, epoch, args, wandb_run=None):
    # Initialize WandB with the provided project name

    # Move the model to the specified device
    net.to(args.device)
    net.train()

    for batch_idx in range(args.n_tasks):
        start_time = time.time()

        # Sample a batch of support and query images and labels
        x_spt, y_spt, x_qry, y_qry, _ = next(db)

        # Move data to the specified device
        x_spt, y_spt = x_spt.to(args.device), y_spt.to(args.device)
        x_qry, y_qry = x_qry.to(args.device), y_qry.to(args.device)

        task_num, setsz, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # Initialize the inner optimizer for adaptation
        inner_opt = torch.optim.SGD(net.parameters(), lr=args.meta_lr)

        qry_losses = []
        qry_accs = []
        args.meta_opt.zero_grad()

        for i in range(task_num):
            # utilize the sec-derivative gradient information for the inner loop
            with higher.innerloop_ctx(
                net, inner_opt, copy_initial_weights=False
            ) as (fnet, diffopt):
                # Inner-loop adaptation
                for _ in range(args.n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i].long())
                    diffopt.step(spt_loss)

                # Compute the loss and accuracy on the query set
                qry_logits = fnet(x_qry[i])
                qry_loss = F.cross_entropy(qry_logits, y_qry[i].long())
                qry_losses.append(qry_loss.detach())

                softmax_qry = F.softmax(qry_logits, dim=1)
                softmax_qry = torch.argmax(softmax_qry, dim=1)
                Dice_score = dice_coefficient_time_series(
                    softmax_qry, y_qry[i].long()
                )

                qry_accs.append(Dice_score)

                # Backpropagation on query loss
                qry_loss.backward()

        args.meta_opt.step()

        qry_losses = sum(qry_losses) / task_num
        qry_accs = 100.0 * sum(qry_accs) / task_num
        i = epoch + float(batch_idx) / args.n_tasks
        iter_time = time.time() - start_time

        if batch_idx % args.log_interval == 0:
            print(
                f"[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f} | Time: {iter_time:.2f}"
            )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/epoch": i,
                    "train/loss": qry_losses,
                    "train/acc": qry_accs,
                    "train/time": time.time(),
                }
            )
    # wandb.finish()
    return


def test(db, net, epoch, args, wandb_r=None):
    # Crucially in our testing procedure here, we do *not* fine-tune
    # the model during testing for simplicity.
    # Most research papers using MAML for this task do an extra
    # stage of fine-tuning here that should be added if you are
    # adapting this code for research.

    net.to(args.device)
    net.train()
    n_test_iter = args.n_tasks

    qry_losses = []
    qry_accs = []

    for batch_idx in range(n_test_iter):
        # Sample a batch of support and query images and labels
        x_spt, y_spt, x_qry, y_qry, _ = next(db)

        # Move data to the specified device
        x_spt, y_spt = x_spt.to(args.device), y_spt.to(args.device)
        x_qry, y_qry = x_qry.to(args.device), y_qry.to(args.device)

        task_num, setsz, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # Initialize the inner optimizer for adaptation
        inner_opt = torch.optim.SGD(net.parameters(), lr=args.meta_lr)

        for i in range(task_num):
            with higher.innerloop_ctx(
                net, inner_opt, track_higher_grads=False
            ) as (fnet, diffopt):
                # Inner-loop adaptation
                for _ in range(args.n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i])
                    diffopt.step(spt_loss)

                # Compute the query loss and accuracy
                qry_logits = fnet(x_qry[i]).detach()
                qry_loss = F.cross_entropy(
                    qry_logits, y_qry[i], reduction="none"
                )
                qry_losses.append(qry_loss.detach())
                softmax_qry = F.softmax(qry_logits, dim=1)
                softmax_qry = torch.argmax(softmax_qry, dim=1)
                Dice_score = dice_coefficient_time_series(
                    softmax_qry, y_qry[i].long()
                )
                qry_accs.append(Dice_score)

    qry_losses = torch.cat(qry_losses).mean().item()
    qry_accs = 100.0 * sum(qry_accs) / task_num / n_test_iter

    print(
        f"[Epoch {epoch + 1:.2f}] Test Loss: {qry_losses:.2f} | Acc: {qry_accs:.2f}"
    )

    if wandb_r is not None:
        # Log results
        wandb_r.log(
            {
                "test/epoch": epoch + 1,
                "test/loss": qry_losses,
                "test/acc": qry_accs,
                "test/time": time.time(),
            }
        )
    return qry_losses, qry_accs



def main(args):
    # Initialize datasets
    train_dataset, test_dataset = get_dataset(args)

    # Initialize samplers
    train_sampler = DenseLabelTaskSampler(
        train_dataset,
        n_shot=args.n_shot,
        batch_size=args.batch_size,
        n_query=args.n_query,
        n_tasks=args.n_tasks,
        threshold_ratio=args.threshold_ratio,
        add_side_noise=args.add_side_noise,
    )
    test_sampler = DenseLabelTaskSampler(
        test_dataset,
        n_shot=args.n_shot,
        batch_size=args.batch_size,
        n_query=args.n_query,
        n_tasks=args.n_tasks,
        threshold_ratio=args.threshold_ratio,
        add_side_noise=args.add_side_noise,
    )

    # Initialize DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=train_sampler.episodic_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    # Define the model architecture
    model = get_model(args)
    
    model.to(args.device)  # Move model to specified device

    # Initialize meta optimizer
    args.meta_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if not args.nowandb:
        run = wandb.init(
            project=args.wandb_project,
            config=vars(args),
        )
        run.name = args.model + "-" + run.name
    else:
        run = None
    # Training loop
    loss = np.inf
    for epoch in range(args.n_epochs):
        # args.epoch = epoch  # Update epoch in args
        train(iter(train_loader), model, epoch, args, run)
        qry_loss, qry_acc = test(iter(test_loader), model, epoch, args, run)
        if qry_loss < loss:
            loss = qry_loss
            torch.save(model.state_dict(), "saved_model/" + run.name + ".pth")
    if not args.nowandb:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file("saved_model/" + run.name + ".pth")
        run.log_artifact(artifact)

        run.finish()
    return


if __name__ == "__main__":
    args = get_args()  # Get arguments from the argparse
    args.wandb_group = "experiment-" + wandb.util.generate_id()
    main(args)
