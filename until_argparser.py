import argparse

from datasets.PhysiQ import PhysiQ
from datasets.Transforms import IMUAugmentation
from methods.EX import EX
from methods.transformer import TransformerModel
from methods.unet import EXACT_UNet, UNet
import torch.nn as nn


def get_model_args(args, preliminary_args):
    model = preliminary_args.model.lower()
    if model == "unet":
        args = UNet.add_args(args)
    elif model == "exact_unet":
        args = EXACT_UNet.add_args(args)
    elif model == "transformer":
        args = TransformerModel.add_args(args)
    elif model == "ex":
        args = EX.add_args(args)
    return args


def get_dataset_args(args, preliminary_args):
    # without parse args to get dataset:
    dataset = preliminary_args.dataset.lower()
    if dataset == "physiq":
        args = PhysiQ.add_args(args)
    return args


def get_model(args):
    args.model = args.model.lower()
    if args.model == "unet":
        model = UNet
    elif args.model == "exact_unet":
        model = EXACT_UNet
    elif args.model == "transformer":
        model = TransformerModel
    elif args.model == "ex":
        model = EX
    else:
        raise ValueError("Model not supported")

    class UNet_wrapper(nn.Module):
        def __init__(self):
            super(UNet_wrapper, self).__init__()
            self.net = model(args)

        def forward(self, x):
            x = x.float()
            x = self.net(x)
            x = x.permute(0, 2, 1)
            return x.squeeze(1)

    # Initialize model
    model = UNet_wrapper().float()
    return model


def get_dataset(args):
    dataset = args.dataset.lower()
    if dataset == "physiq":
        train_dataset = PhysiQ(
            root=args.data_root,
            split="train",
            window_size=args.window_size,
            bg_fg=None,
            args=args,
            transforms=IMUAugmentation(rotation_chance=0),
        )
        test_dataset = PhysiQ(
            root=args.data_root,
            split="test",
            window_size=args.window_size,
            bg_fg=None,
            args=args,
        )
    else:
        raise ValueError("Dataset not supported")
    return train_dataset, test_dataset


def get_args():
    parser = argparse.ArgumentParser(
        description="Meta-learning for dense labeling tasks"
    )

    # Dataset and DataLoader parameters
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory for the dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="physiq",
        help="Dataset to use for training",
    )
    
    """Add dataset-specific arguments to the parser."""
    parser.add_argument(
        "--shuffle",
        type=str,
        default="random",
        help="Shuffle the data on each subject",
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=42,
        help="Seed for shuffling the data",
    )
    parser.add_argument(
        "--add_side_noise",
        help="Add random noise to the data on each side",
        action="store_true",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="white",
        help="Type of noise to add to the data",
    )
    
    
    parser.add_argument(
        "--in_channels",
        type=int,
        default=6,
        help="Input channels for the model",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=2,
        help="Output channels for the model",
    )
    
    parser.add_argument(
        "--window_size",
        type=int,
        default=200,
        help="Window size for the PhysiQ dataset",
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=1,
        help="Number of support examples per class",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--n_query",
        type=int,
        default=1,
        help="Number of query examples per class",
    )
    parser.add_argument(
        "--n_tasks", type=int, default=1000, help="Number of tasks for training"
    )
    parser.add_argument(
        "--threshold_ratio",
        type=float,
        default=0.25,
        help="Threshold ratio for DenseLabelTaskSampler",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Use pinned memory for DataLoader",
    )

    # Training parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the meta optimizer",
    )
    parser.add_argument(
        "--meta_lr",
        type=float,
        default=1e-2,
        help="Learning rate for the inner optimizer",
    )
    parser.add_argument(
        "--n_inner_iter",
        type=int,
        default=1,
        help="Number of inner-loop iterations",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=4,
        help="Interval for logging training metrics",
    )
    
    # WandB parameters
    parser.add_argument(
        "--wandb_project", type=str, default="EXACT", help="WandB project name"
    )
    parser.add_argument(
            "--nowandb",
            action="store_true",
        )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="unet",
        help="Model to use for training",
    )
    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument("--dataset", type=str, default="physiq", help="Specify the dataset")
    preliminary_parser.add_argument("--model", type=str, default="unet", help="Specify the model")
    # Parse known arguments to extract the dataset
    preliminary_args, _ = preliminary_parser.parse_known_args()

    # Add PhysiQ-specific arguments
    parser = get_dataset_args(parser, preliminary_args)
    parser = get_model_args(parser, preliminary_args)
    # model = get_model(parser.parse_args())

    return parser.parse_args()
