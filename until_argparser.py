import argparse

from datasets.PhysiQ import PhysiQ
from methods.transformer import TransformerModel
from methods.unet import EXACT_UNet, UNet


def get_model_args(args):
    model = args.parse_args().model.lower()
    if model == "unet":
        args = UNet.add_args(args)
    elif model == "exact_unet":
        args = EXACT_UNet.add_args(args)
    elif model == "transformer":
        args = TransformerModel.add_args(args)
    return args


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
        "--n_train_iter",
        type=int,
        default=1000,
        help="Number of training iterations per epoch",
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
        "-m",
        "--model",
        type=str,
        default="unet",
        help="Model to use for training",
    )

    # Add PhysiQ-specific arguments
    parser = PhysiQ.add_args(parser)
    parser = get_model_args(parser)
    # model = get_model(parser.parse_args())

    return parser.parse_args()
