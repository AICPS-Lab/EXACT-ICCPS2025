import argparse
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import necessary modules from your project
from until_argparser import get_args, get_dataset, get_model
from datasets.DenseLabelTaskSampler import DenseLabelTaskSampler
from loss_fn import MetricsAccumulator

import wandb

def load_model_from_wandb(args):
    # run = wandb.init(project=args.wandb_project, 
    #                  name =f"{args.model}-{args.dataset}-{str(args.seed)}")

    run = wandb.init()


    artifact = run.use_artifact('entity/your-project-name/model:v0', type='model')
    artifact_dir = artifact.download()


    run.finish()


    artifact = run.use_artifact(f'model:latest')
    artifact_dir = artifact.download()
    model_path = os.path.join(artifact_dir, 'model.pth')
    model = get_model(args)
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)
    model.eval()
    return model


def prepare_data(args):
    _, test_dataset = get_dataset(args)
    test_sampler = DenseLabelTaskSampler(
        test_dataset,
        n_shot=args.n_shot,
        batch_size=args.batch_size,
        n_query=args.n_query,
        n_tasks=args.n_tasks,
        threshold_ratio=args.threshold_ratio,
        add_side_noise=args.add_side_noise,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=test_sampler.episodic_collate_fn,
    )
    return test_loader

def evaluate(model, test_loader, args):
    compute_metrics = MetricsAccumulator(dir_name="evaluation")
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (x_spt, y_spt, x_qry, y_qry, _) in enumerate(test_loader):
            x_qry, y_qry = x_qry.to(args.device), y_qry.to(args.device)
            # Assuming task_num is 1 for simplicity
            qry_logits = model(x_qry[0])
            # Collect predictions and labels for visualization
            all_predictions.append(qry_logits.cpu())
            all_labels.append(y_qry[0].cpu())
            # Update metrics
            compute_metrics.update(y_qry[0].long(), qry_logits)

    # Compute final metrics
    res_dict = compute_metrics.compute()
    print("Evaluation Metrics:", res_dict)
    return all_predictions, all_labels



def main():
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_from_wandb(args)
    test_loader = prepare_data(args)
    all_predictions, all_labels = evaluate(model, test_loader, args)
    # visualize_results(all_predictions, all_labels)

if __name__ == '__main__':
    main()
