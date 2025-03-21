import argparse
import os
import time
import typing

import higher
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

import wandb
from datasets.DenseLabelTaskSampler import DenseLabelTaskSampler
from MetricsAccumulator import MetricsAccumulator, MetricsAccumulator_v2
from methods import EX, UNet
from until_argparser import get_all_subjects, get_args, get_dataset, get_model
from utilities import model_exception_handler, printc, seed
from utils_metrics import fsl_visualize_softmax, visualize_softmax


def train(db, net, epoch, args, wandb_run=None):
    # Initialize WandB with the provided project name

    # Move the model to the specified device
    net.to(args.device)
    net.train()
    compute_metrics = MetricsAccumulator(dir_name="train", n_classes=args.out_channels)

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
        args.meta_opt.zero_grad()

        for i in range(task_num):
            # utilize the sec-derivative gradient information for the inner loop
            with higher.innerloop_ctx(
                net, inner_opt, copy_initial_weights=False, track_higher_grads=True
            ) as (fnet, diffopt):
                # Inner-loop adaptation
                for _ in range(args.n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i].long())
                    # diffopt.step(spt_loss)
                    # diffopt.step(spt_loss, grad_callback=lambda grads: [g.detach() for g in grads])
                    if args.fomaml:
                        diffopt.step(spt_loss, grad_callback=lambda grads: [g.detach() if g is not None else None for g in grads])
                    else:
                        diffopt.step(spt_loss)

                # Compute the loss and accuracy on the query set
                qry_logits = fnet(x_qry[i])
                qry_loss = F.cross_entropy(qry_logits, y_qry[i].long())
                qry_losses.append(qry_loss.detach())
                string_score, score = compute_metrics.update(
                    y_qry[i].long(), qry_logits
                )
                # Backpropagation on query loss
                qry_loss.backward()

        args.meta_opt.step()

        qry_losses = sum(qry_losses) / task_num
        i = epoch + float(batch_idx) / args.n_tasks
        iter_time = time.time() - start_time

        if wandb_run is None and batch_idx % args.log_interval == 0:
            print(
                f"[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Time: {iter_time:.2f}",
                end=" | ",
            )
            print(string_score)

        res_dict = compute_metrics.compute()
        res_dict.update(
            {
                "train/epoch": i,
                "train/loss": qry_losses,
                "train/time": time.time(),
            }
        )
        if wandb_run is not None:
            wandb_run.log(res_dict)
        compute_metrics.reset()
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

    
    # if args.loocv:
    #     compute_metrics = MetricsAccumulator_v2(dir_name="test", n_classes=args.out_channels)
    # else:
    compute_metrics = MetricsAccumulator(dir_name="test", n_classes=args.out_channels)

    qry_losses = []

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
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i].long())
                    diffopt.step(spt_loss)

                # Compute the query loss and accuracy
                qry_logits = fnet(x_qry[i]).detach()
                qry_loss = F.cross_entropy(
                    qry_logits, y_qry[i].long(), reduction="none"
                )
                qry_losses.append(qry_loss.detach())
                string_score, score = compute_metrics.update(
                    y_qry[i].long(), qry_logits
                )

    qry_losses = torch.cat(qry_losses).mean().item()

    # print(f"[Epoch {epoch + 1:.2f}] Test Loss: {qry_losses:.2f}", end=" | ")
    # print(string_score)

    res_dict = compute_metrics.compute()
    res_dict.update(
        {
            "test/epoch": epoch + 1,
            "test/loss": qry_losses,
            "test/time": time.time(),
        }
    )
    if wandb_r is not None:
        # Log results
        wandb_r.log(res_dict)
        log_visualization(epoch, wandb_r,net, fnet, inner_opt, args)
    return qry_losses, res_dict

def main(args):
    # Initialize datasets
    train_dataset, test_dataset = get_dataset(args)
    seed(args.seed)
    # Initialize samplers
    train_sampler = DenseLabelTaskSampler(
        train_dataset,
        n_shot=args.n_shot,
        batch_size=args.batch_size,
        n_query=args.n_query,
        n_tasks=args.n_tasks,
        threshold_ratio=args.threshold_ratio,
        add_side_noise=args.add_side_noise,
        args=args
    )
    test_sampler = DenseLabelTaskSampler(
        test_dataset,
        n_shot=args.n_shot,
        batch_size=args.batch_size,
        n_query=args.n_query,
        n_tasks=args.n_tasks,
        threshold_ratio=args.threshold_ratio,
        add_side_noise=args.add_side_noise,
        args=args
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

    # save 10 samples of data for visualization 
    capture_test_dataset_samples(args, test_dataset, test_loader)
    
    # Define the model architecture
    model = get_model(args)

    model.to(args.device)  # Move model to specified device
    
    # Initialize meta optimizer
    args.meta_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if not args.nowandb:
        run = wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name= f"{args.model}-{args.dataset}-{str(args.seed)}"
        )
        run.name = run.name
        model_path = "saved_model/" + run.name + ".pth"
    else:
        run = None
        model_path = "saved_model/" + args.model + ".pth"
    model_exception_handler(model_path)

    # Training loop
    loss = np.inf
    for epoch in range(args.n_epochs):
        # args.epoch = epoch  # Update epoch in args
        train(iter(train_loader), model, epoch, args, run)
        qry_loss, qry_acc = test(iter(test_loader), model, epoch, args, run)
        if qry_loss < loss:
            loss = qry_loss
            torch.save(model.state_dict(), model_path)
    if not args.nowandb:
        log_model_artifact(run, model_path)
        run.finish()
    return

def main_loocv(args):
    # Set up for each subject as a separate LOOCV iteration
    all_subjects = get_all_subjects(args)  # Define a function to get all subject IDs

    for test_subject in all_subjects: # Iterate over first 10 subjects
        # Initialize datasets with the current subject as test
        train_dataset, test_dataset = get_dataset(args, test_subject)
        seed(args.seed)
        
        # Initialize samplers
        train_sampler = DenseLabelTaskSampler(
            train_dataset,
            n_shot=args.n_shot,
            batch_size=args.batch_size,
            n_query=args.n_query,
            n_tasks=args.n_tasks,
            threshold_ratio=args.threshold_ratio,
            add_side_noise=args.add_side_noise,
            args=args
        )
        test_sampler = DenseLabelTaskSampler(
            test_dataset,
            n_shot=args.n_shot,
            batch_size=args.batch_size,
            n_query=args.n_query,
            n_tasks=args.n_tasks,
            threshold_ratio=args.threshold_ratio,
            add_side_noise=args.add_side_noise,
            args=args
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
        capture_test_dataset_samples(args, test_dataset, test_loader)
        # Define the model architecture
        model = get_model(args)
        model.to(args.device)

        print(
            "trainable parameters: ",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

        # Initialize meta optimizer
        args.meta_opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Run a unique wandb session for each test subject
        if not args.nowandb:
            run = wandb.init(
                project=args.wandb_project,
                config=vars(args),
                name=f"{args.model}-{args.dataset}-{args.seed}-subject-{test_dataset.test_subject_filename()}",
            )
            model_path = f"saved_model/{run.name}.pth"
        else:
            run = None
            model_path = f"saved_model/{args.model}_subject_{test_dataset.test_subject_filename()}.pth"
        
        model_exception_handler(model_path)

        # Training loop
        loss = np.inf
        for epoch in range(args.n_epochs):
            train(iter(train_loader), model, epoch, args, run)
            qry_loss, qry_acc = test(iter(test_loader), model, epoch, args, run)
            if qry_loss < loss:
                loss = qry_loss
                torch.save(model.state_dict(), model_path)
        
        # Save the model to wandb for each run
        if not args.nowandb:
            log_model_artifact(run, model_path)
            run.finish()
        
    return

def log_model_artifact(run, model_path):
    artifact = wandb.Artifact(f"{run.name}", type="model")
    artifact.add_file(model_path)
    run.log_artifact(artifact)

def capture_test_dataset_samples(args, test_dataset, test_loader):
    # if it exists already:
    if os.path.exists(os.path.join(args.data_root, args.dataset, f"{args.dataset}_{str(args.loocv)}_{args.seed}.pt")):
        # printc("Saved 10 samples data already exists")
        return
    sample_data_in_test = []
    for i in range(10):
        x_spt, y_spt, x_qry, y_qry, _ = next(iter(test_loader))
        sample_data_in_test.append((x_spt, y_spt, x_qry, y_qry))
    # torch save
    torch.save(sample_data_in_test, os.path.join(args.data_root, args.dataset, f"{args.dataset}_{str(args.loocv)}_{args.seed}.pt"))
    # printc("Saved 10 samples of data for visualization")

def log_visualization(epoch, wandb_r, net, fnet, inner_opt, args):
    # load 10 samples of data for visualization
    sample_data_in_test = torch.load(os.path.join(args.data_root, args.dataset, f"{args.dataset}_{str(args.loocv)}_{args.seed}.pt"))
    for image_idx, (x_spt, y_spt, x_qry, y_qry) in enumerate(sample_data_in_test):
        # visualize the softmax output of the model
        x_spt, y_spt = x_spt.to(args.device), y_spt.to(args.device)
        x_qry, y_qry = x_qry.to(args.device), y_qry.to(args.device)
        with higher.innerloop_ctx(
                net, inner_opt, track_higher_grads=False
            ) as (fnet, diffopt):
                # Inner-loop adaptation
                for _ in range(args.n_inner_iter):
                    spt_logits = fnet(x_spt[:, 0])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[:, 0].long())
                    if args.fomaml:
                        diffopt.step(spt_loss, grad_callback=lambda grads: [g.detach() if g is not None else None for g in grads])
                    else:
                        diffopt.step(spt_loss)
        images = x_qry[:, 0]
        qry_logits = fnet(images).detach()
        labels = y_qry[:, 0]
        softmaxed = F.softmax(qry_logits, dim=1)
        fsl_visualize_softmax(
            softmaxed[0].cpu().detach().numpy(),
            labels[0].cpu().detach().numpy(),
            images[0].cpu().detach().numpy(),
            x_spt[0, 0].cpu().detach().numpy(),  # Support set data
            y_spt[0, 0].cpu().detach().numpy(),  # Support set labels
            dir=f"saved_figure/{wandb_r.name}_epoch_{epoch}_img_{image_idx}.png"
        )
        # log a visualization: 
        # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb#scrollTo=hPqfetlsyXbg
        wandb_r.log({f"img_{image_idx}": wandb.Image(f"saved_figure/{wandb_r.name}_epoch_{epoch}_img_{image_idx}.png")})
        os.remove(f"saved_figure/{wandb_r.name}_epoch_{epoch}_img_{image_idx}.png")
        
    # images = x_qry[0]
    # qry_logits = fnet(images).detach()
    # labels = y_qry[0]
    # softmaxed = F.softmax(qry_logits, dim=1)
    # visualize_softmax(softmaxed[i].cpu().detach().numpy(), labels[i].cpu().detach().numpy(), images[i].cpu().detach().numpy(), dir=f"saved_figure/{wandb_r.name}_epoch_{epoch}.png")
    #     # log a visualization: 
    #     # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb#scrollTo=hPqfetlsyXbg
    # wandb_r.log({"img": wandb.Image(f"saved_figure/{wandb_r.name}_epoch_{epoch}.png")})
    # os.remove(f"saved_figure/{wandb_r.name}_epoch_{epoch}.png")

if __name__ == "__main__":
    args = get_args()  # Get arguments from the argparse
    args.wandb_group = "experiment-" + wandb.util.generate_id()
    if args.loocv:
        main_loocv(args)
    else:
        main(args)
