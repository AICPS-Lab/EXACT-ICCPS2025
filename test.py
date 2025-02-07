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
from MetricsAccumulator import MetricsAccumulator
from main_meta_v2 import capture_test_dataset_samples
from methods import EX, UNet
from until_argparser import get_all_subjects, get_args, get_dataset, get_model
from utilities import model_exception_handler, printc, seed
from utils_metrics import fsl_visualize_softmax, visualize_softmax
import main_meta_v2

def main_test_only(args):
    train_dataset, test_dataset = get_dataset(args)
    seed(args.seed)
    # Initialize samplers
    
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
    run = None
    model_path = f"saved_model/wandb_artifacts/{args.model}_{args.dataset}/{args.model}-{args.dataset}-{args.seed}.pth"
    # model_exception_handler(model_path)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    epoch = 200
    # if model_path exists:
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist.")
        return
    qry_loss, qry_acc = main_meta_v2.test(iter(test_loader), model, epoch, args, run)

    # print(f"Test Loss: {qry_loss:.4f}, qry_acc: {qry_acc}")
    return qry_acc

def main_loocv_test_only(args, filename, test_subject):
    # Set up for each subject as a separate LOOCV iteration
    all_subjects = get_all_subjects(args)  # Define a function to get all subject IDs

    
    _, test_dataset = get_dataset(args, test_subject)
    seed(args.seed)
    
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

    # Initialize meta optimizer
    args.meta_opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Run a unique wandb session for each test subject
    run = None
    model_path = filename
    

    epoch = 200
    qry_loss, qry_acc = main_meta_v2.test(iter(test_loader), model, epoch, args, run)
        
    return


if __name__ == "__main__":
    args = get_args()  # Get arguments from the argparse
    args.nowandb = True
    if args.loocv:
        # read the file from the path:
        path = f"saved_model/wandb_artifacts/subjects/{args.model}_{args.dataset}/"
        # get all pth files in the path
        files = [f for f in os.listdir(path) if f.endswith(".pth")]
        iou = []
        dice = []
        rocauc = []
        for filename in files:
            # get the path of the file
            filename = os.path.join(path, filename)
            subjects = filename.split("/")[-1].split("-")[-1].split("_")
            print()
            main_loocv_test_only(args, filename, subjects)
        
        
    else: 
        args.add_side_noise = True
        # if cuda is available, use it
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # args.dataset ='mmfit'
        iou = []
        dice = []
        rocauc = []
        for iseed in range(42, 47):
            args.seed = iseed
            res = main_test_only(args)
            iou.append(res['test/IoU'])
            dice.append(res['test/Dice Score'])
            rocauc.append(res['test/ROC-AUC'])
        print("DICE | IOU | ROC-AUC")
        # with 3 decimal points
        print(args.model, f"{np.mean(dice):.3f} ± {np.std(dice):.4f} | {np.mean(iou):.4f} ± {np.std(iou):.4f} | {np.mean(rocauc):.3f} ± {np.std(rocauc):.4f}")
    
