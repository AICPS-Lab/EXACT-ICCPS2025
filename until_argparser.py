import argparse

from datasets.PhysiQ import PhysiQ


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

    # Model parameters
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
        "--epochs", type=int, default=100, help="Number of training epochs"
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
    
    # Add PhysiQ-specific arguments
    parser = PhysiQ.add_args(parser)

    return parser.parse_args()
