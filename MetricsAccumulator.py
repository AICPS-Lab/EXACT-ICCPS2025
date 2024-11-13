from collections import Counter
from typing import Counter, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.nn import functional as F

# loss fn and eval metrics:


class MetricsAccumulator:
    def __init__(self, dir_name=None, n_classes=3):
        self.dir_name = dir_name
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        # Accumulators for counts
        self.class_tp = np.zeros(self.n_classes)
        self.class_fp = np.zeros(self.n_classes)
        self.class_fn = np.zeros(self.n_classes)
        self.class_tn = np.zeros(self.n_classes)

        # Accumulators for ROC-AUC
        self.all_y_true = []
        self.all_y_probs = []  # Accumulator for probabilities

        # For Dice and IoU
        self.class_intersection = np.zeros(self.n_classes)
        self.class_union = np.zeros(self.n_classes)

        # Total number of samples
        self.total_samples = 0

    def update(self, y_true, logits) -> Tuple[str, dict]:
        if torch.isnan(logits).any():
            print("NaNs detected in logits")
        # Ensure inputs are torch tensors
        if not torch.is_tensor(y_true):
            y_true = torch.tensor(y_true)
        if not torch.is_tensor(logits):
            logits = torch.tensor(logits)

        # Check and adjust logits dimensions
        assert (
            len(y_true.size()) == 2
        ), f"y_true size: {y_true.size()} is not compatible"
        if logits.size()[-1] != self.n_classes:
            logits = logits.permute(0, 2, 1)

        if y_true.size() == logits.size()[:-1]:
            assert (
                len(logits.size()) == 3
            ), f"logits size: {logits.size()} is not compatible"
            probs = F.softmax(logits, dim=2)
            y_pred = torch.argmax(probs, dim=2)
        elif y_true.size() == logits.size():
            raise ValueError("Please provide logits before softmax")
        else:
            raise ValueError(
                f"y_true size: {y_true.size()} and logits size: {logits.size()} are not compatible"
            )

        # Flatten tensors for batch processing
        y_true_flat = y_true.view(-1)
        y_pred_flat = y_pred.view(-1)
        probs_flat = probs.view(-1, self.n_classes)

        y_true_np = y_true_flat.detach().cpu().numpy()
        y_pred_np = y_pred_flat.detach().cpu().numpy()
        probs_np = probs_flat.detach().cpu().numpy()

        # Accumulate counts
        cm = confusion_matrix(
            y_true_np, y_pred_np, labels=range(self.n_classes)
        )
        self.class_tp += np.diag(cm)
        self.class_fp += cm.sum(axis=0) - np.diag(cm)
        self.class_fn += cm.sum(axis=1) - np.diag(cm)
        self.class_tn += cm.sum() - (
            self.class_fp + self.class_fn + self.class_tp
        )

        # Accumulate predictions and labels for ROC-AUC
        self.all_y_true.extend(y_true_np)
        self.all_y_probs.append(probs_np)  # Accumulate probabilities

        # Accumulate for Dice and IoU
        for class_idx in range(self.n_classes):
            y_true_class = (y_true_flat == class_idx).float()
            y_pred_class = (y_pred_flat == class_idx).float()

            intersection = (y_true_class * y_pred_class).sum().item()
            union = y_true_class.sum().item() + y_pred_class.sum().item()

            self.class_intersection[class_idx] += intersection
            self.class_union[class_idx] += union

        # Update total samples
        self.total_samples += y_true_flat.numel()

        # Compute batch-specific metrics for immediate reporting
        precision = precision_score(
            y_true_np, y_pred_np, average="macro", zero_division=0
        )
        recall = recall_score(
            y_true_np, y_pred_np, average="macro", zero_division=0
        )
        f1 = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)

        # Dice and IoU
        dice_scores = (2 * self.class_intersection) / (self.class_union + 1e-7)
        mean_dice = np.mean(dice_scores)
        iou_scores = self.class_intersection / (
            self.class_union - self.class_intersection + 1e-7
        )
        mean_iou = np.mean(iou_scores)

        # ROC-AUC
        if self.n_classes == 2:
            y_scores = probs_np[:, 1]  # Probabilities for the positive class
            y_true_binary = y_true_np
            if len(np.unique(y_true_binary)) == 2:
                try:
                    roc_auc = roc_auc_score(y_true_binary, y_scores)
                except ValueError:
                    roc_auc = float("nan")
            else:
                roc_auc = float("nan")
        else:
            y_true_binarized = label_binarize(
                y_true_np, classes=range(self.n_classes)
            )
            if len(np.unique(y_true_np)) >= 2:
                try:
                    roc_auc = roc_auc_score(
                        y_true_binarized, probs_np, multi_class="ovr"
                    )
                except ValueError:
                    roc_auc = float("nan")
            else:
                roc_auc = float("nan")

        # Create a summary string of current batch metrics
        batch_metrics = (
            f"Prec: {precision:.2f}, Rec: {recall:.2f}, "
            f"F1: {f1:.2f}, Dice: {mean_dice:.2f}, "
            f"IoU: {mean_iou:.2f}, ROC-AUC: {roc_auc:.2f}"
        )
        batch_dict = {
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Dice Score": mean_dice,
            "IoU": mean_iou,
            "ROC-AUC": roc_auc,
        }

        # Return the string summary and the dict of the results
        return batch_metrics, self._dict_format(batch_dict)

    def _dict_format(self, metrics) -> dict:
        if self.dir_name is not None:
            return {
                f"{self.dir_name}/{key}": value
                for key, value in metrics.items()
            }
        else:
            return metrics

    def compute(self) -> dict:
        if self.total_samples == 0:
            raise ValueError("No data to compute metrics. Call update() first.")

        # Precision, Recall, F1-Score
        precision = np.mean(
            self.class_tp / (self.class_tp + self.class_fp + 1e-7)
        )
        recall = np.mean(self.class_tp / (self.class_tp + self.class_fn + 1e-7))
        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        # Dice Score and IoU
        dice_scores = (2 * self.class_intersection) / (self.class_union + 1e-7)
        mean_dice = np.mean(dice_scores)
        iou_scores = self.class_intersection / (
            self.class_union - self.class_intersection + 1e-7
        )
        mean_iou = np.mean(iou_scores)

        # ROC-AUC
        all_y_true_np = np.array(self.all_y_true)
        all_y_probs_np = np.vstack(self.all_y_probs)

        if self.n_classes == 2:
            y_scores = all_y_probs_np[
                :, 1
            ]  # Probabilities for the positive class
            y_true_binary = all_y_true_np
            if len(np.unique(y_true_binary)) == 2:
                try:
                    roc_auc = roc_auc_score(y_true_binary, y_scores)
                except ValueError:
                    roc_auc = float("nan")
            else:
                roc_auc = float("nan")
        else:
            y_true_binarized = label_binarize(
                all_y_true_np, classes=range(self.n_classes)
            )
            if len(np.unique(all_y_true_np)) >= 2:
                try:
                    roc_auc = roc_auc_score(
                        y_true_binarized, all_y_probs_np, multi_class="ovr"
                    )
                except ValueError:
                    roc_auc = float("nan")
            else:
                roc_auc = float("nan")

        metrics = {
            "Dice Score": mean_dice,
            "IoU": mean_iou,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc,
        }
        return self._dict_format(metrics)


def test_metrics_accumulator():
    # Case 1: Both classes (0 and 1) are present in y_true
    print("Test Case 1: Both classes present")
    y_true_1 = torch.tensor([[1, 0, 1, 1, 0]])
    logits_1 = torch.tensor(
        [[[0.2, 0.8], [0.9, 0.1], [0.3, 0.7], [0.4, 0.6], [0.6, 0.4]]]
    )

    accumulator_1 = MetricsAccumulator()
    batch_metrics_1, _ = accumulator_1.update(y_true_1, logits_1)
    computed_metrics_1 = accumulator_1.compute()

    print("Batch metrics:", batch_metrics_1)
    print("Computed metrics:", computed_metrics_1)
    print()

    # Case 2: Only one class (1) present in y_true
    print("Test Case 2: Only class 1 present")
    y_true_2 = torch.tensor([[1, 1, 1, 1, 1]])
    logits_2 = torch.tensor(
        [[[0.2, 0.8], [0.3, 0.7], [0.1, 0.9], [0.4, 0.6], [0.3, 0.7]]]
    )

    accumulator_2 = MetricsAccumulator()
    batch_metrics_2, _ = accumulator_2.update(y_true_2, logits_2)
    try:
        computed_metrics_2 = accumulator_2.compute()
    except ValueError as e:
        computed_metrics_2 = str(e)

    print("Batch metrics:", batch_metrics_2)
    print("Computed metrics:", computed_metrics_2)
    print()

    # Case 3: Only one class (0) present in y_true
    print("Test Case 3: Only class 0 present")
    y_true_3 = torch.tensor([[0, 0, 0, 0, 0]])
    logits_3 = torch.tensor(
        [[[0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [0.85, 0.15], [0.75, 0.25]]]
    )

    accumulator_3 = MetricsAccumulator()
    batch_metrics_3, _ = accumulator_3.update(y_true_3, logits_3)
    batch_metrics_3, _ = accumulator_3.update(y_true_3, logits_3)
    batch_metrics_3, _ = accumulator_3.update(y_true_3, logits_3)
    batch_metrics_3, _ = accumulator_3.update(y_true_2, logits_2)
    try:
        computed_metrics_3 = accumulator_3.compute()
    except ValueError as e:
        computed_metrics_3 = str(e)

    print("Batch metrics:", batch_metrics_3)
    print("Computed metrics:", computed_metrics_3)
    print()


if __name__ == "__main__":
    # Example usage
    metrics_accumulator = MetricsAccumulator()

    # Replace with batches of data in actual usage
    y_true = torch.tensor(
        [[0, 1, 1, 0, 1], [0, 0, 1, 1, 0]]
    )  # Ground truth labels (batch of 2 sequences)
    logits = torch.tensor(
        [  # Model outputs (logits)
            [[2.0, 1.0], [0.5, 2.5], [1.5, 1.0], [2.0, 0.5], [0.0, 3.0]],
            [[1.0, 1.0], [2.5, 0.5], [0.5, 2.0], [1.0, 1.0], [2.0, 0.5]],
        ]
    )

    # Simulate updating over multiple batches
    string = metrics_accumulator.update(y_true, logits)
    print(string)
    y_true = torch.tensor(
        [[0, 1, 1, 0, 1]]
    )  # Ground truth labels (batch of 2 sequences)
    logits = torch.tensor(
        [  # Model outputs (logits)
            [[2.0, 1.0], [0.5, 2.5], [1.5, 1.0], [2.0, 0.5], [0.0, 3.0]],
        ]
    )
    string = metrics_accumulator.update(
        y_true, logits
    )  # Call update again as if it's another batch
    print(string)
    # Compute and print average metrics
    average_metrics = metrics_accumulator.compute()
    for name, value in average_metrics.items():
        print(f"{name}: {value:.4f}")

    test_metrics_accumulator()
