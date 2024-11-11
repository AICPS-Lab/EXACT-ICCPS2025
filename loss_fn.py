from typing import Counter, Tuple
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

# loss fn and eval metrics:

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize
from typing import Tuple
from collections import Counter

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
        cm = confusion_matrix(y_true_np, y_pred_np, labels=range(self.n_classes))
        self.class_tp += np.diag(cm)
        self.class_fp += cm.sum(axis=0) - np.diag(cm)
        self.class_fn += cm.sum(axis=1) - np.diag(cm)
        self.class_tn += cm.sum() - (self.class_fp + self.class_fn + self.class_tp)

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
            y_true_np, y_pred_np, average='macro', zero_division=0
        )
        recall = recall_score(
            y_true_np, y_pred_np, average='macro', zero_division=0
        )
        f1 = f1_score(
            y_true_np, y_pred_np, average='macro', zero_division=0
        )

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
                    roc_auc = float('nan')
            else:
                roc_auc = float('nan')
        else:
            y_true_binarized = label_binarize(
                y_true_np, classes=range(self.n_classes)
            )
            if len(np.unique(y_true_np)) >= 2:
                try:
                    roc_auc = roc_auc_score(
                        y_true_binarized, probs_np, multi_class='ovr'
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
                f"{self.dir_name}/{key}": value for key, value in metrics.items()
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
        recall = np.mean(
            self.class_tp / (self.class_tp + self.class_fn + 1e-7)
        )
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
            y_scores = all_y_probs_np[:, 1]  # Probabilities for the positive class
            y_true_binary = all_y_true_np
            if len(np.unique(y_true_binary)) == 2:
                try:
                    roc_auc = roc_auc_score(y_true_binary, y_scores)
                except ValueError:
                    roc_auc = float('nan')
            else:
                roc_auc = float('nan')
        else:
            y_true_binarized = label_binarize(
                all_y_true_np, classes=range(self.n_classes)
            )
            if len(np.unique(all_y_true_np)) >= 2:
                try:
                    roc_auc = roc_auc_score(
                        y_true_binarized, all_y_probs_np, multi_class='ovr'
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
    logits_1 = torch.tensor([[[0.2, 0.8], [0.9, 0.1], [0.3, 0.7], [0.4, 0.6], [0.6, 0.4]]])
    
    accumulator_1 = MetricsAccumulator()
    batch_metrics_1, _ = accumulator_1.update(y_true_1, logits_1)
    computed_metrics_1 = accumulator_1.compute()
    
    print("Batch metrics:", batch_metrics_1)
    print("Computed metrics:", computed_metrics_1)
    print()

    # Case 2: Only one class (1) present in y_true
    print("Test Case 2: Only class 1 present")
    y_true_2 = torch.tensor([[1, 1, 1, 1, 1]])
    logits_2 = torch.tensor([[[0.2, 0.8], [0.3, 0.7], [0.1, 0.9], [0.4, 0.6], [0.3, 0.7]]])
    
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
    logits_3 = torch.tensor([[[0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [0.85, 0.15], [0.75, 0.25]]])
    
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

# Example usage with batched data:
# Replace these tensors with your actual time series data


def mse_loss(y_pred, y_true):
    return torch.sum((y_pred - y_true) ** 2)


def bce_loss(y_pred, y_true):

    return F.binary_cross_entropy_with_logits(y_pred, y_true.float())


def smooth_l1_loss(y_pred, y_true):
    abs_diff = torch.abs(y_pred - y_true)
    loss = torch.where(abs_diff < 1, 0.5 * abs_diff**2, abs_diff - 0.5)
    return loss.mean()


from torch import nn


class AdjustSmoothL1Loss(nn.Module):
    # https://github.com/chengyangfu/retinamask/blob/master/maskrcnn_benchmark/layers/adjust_smooth_l1_loss.py
    def __init__(self, num_features, momentum=0.1, beta=1.0 / 9):
        # num_features: it is the output feature size (i.e. main_length, not num of channels)
        super(AdjustSmoothL1Loss, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.beta = beta
        self.register_buffer(
            "running_mean", torch.empty(num_features).fill_(beta)
        )
        self.register_buffer("running_var", torch.zeros(num_features))

    def forward(self, inputs, target, size_average=True):

        n = torch.abs(inputs - target)
        with torch.no_grad():
            if torch.isnan(n.var(dim=0)).sum().item() == 0:
                self.running_mean = self.running_mean.to(n.device)
                self.running_mean *= 1 - self.momentum
                self.running_mean += self.momentum * n.mean(dim=0)
                self.running_var = self.running_var.to(n.device)
                self.running_var *= 1 - self.momentum
                self.running_var += self.momentum * n.var(dim=0)

        beta = self.running_mean - self.running_var
        beta = beta.clamp(max=self.beta, min=1e-3)

        beta = beta.view(-1, self.num_features).to(n.device)
        cond = n < beta.expand_as(n)
        loss = torch.where(cond, 0.5 * n**2 / beta, n - 0.5 * beta)
        if size_average:
            return loss.mean()
        return loss.sum()


class SelfAdjustedSmoothL1Loss(nn.Module):
    def __init__(self, momentum=0.9):
        super(SelfAdjustedSmoothL1Loss, self).__init__()
        self.sigma = 1.0
        self.momentum = momentum

    def forward(self, y_pred, y_true):
        return self.loss(y_pred, y_true)

    def loss(self, y_pred, y_true):
        diff = y_pred - y_true
        abs_diff = torch.abs(diff)

        with torch.no_grad():
            # Calculate the adaptive parameter
            current_inv_sigma2 = torch.median(abs_diff) / 0.6745  # MAD constant
            current_sigma2 = 1.0 / (current_inv_sigma2 * current_inv_sigma2)
            # Momentum-based update of sigma
            self.sigma = (
                self.momentum * self.sigma
                + (1.0 - self.momentum) * current_sigma2
            )

        less_than_one = (abs_diff * abs_diff < self.sigma).float()

        loss = less_than_one * 0.5 * self.sigma * abs_diff * abs_diff + (
            1.0 - less_than_one
        ) * (abs_diff - 0.5 / torch.sqrt(self.sigma))

        return loss.sum()


def shannon_entropy(logits):
    prob = F.softmax(logits, dim=1)
    log_prob = F.log_softmax(logits, dim=1)
    ent = -torch.sum(prob * log_prob, dim=1)
    return ent.mean()


# -------------------------------------------------------------------------------------------------------------------------------------
def iou_time_series(pred, gt):
    # Compute element-wise minimum
    intersection = torch.sum(torch.min(pred, gt), dim=1)

    # Compute element-wise maximum
    union = torch.sum(torch.max(pred, gt), dim=1)

    # Avoid division by zero by adding a small epsilon
    eps = 1e-6
    iou = intersection / (union + eps)

    # Average over the batch dimension
    avg_iou = torch.mean(iou)

    return avg_iou


def recall_time_series(pred, gt, threshold=0.5):
    # Convert to binary form using threshold
    pred_binary = (pred > threshold).float()
    gt_binary = (gt > threshold).float()

    # Compute True Positives
    TP = torch.sum(pred_binary * gt_binary, dim=1)

    # Compute False Negatives
    FN = torch.sum((1 - pred_binary) * gt_binary, dim=1)

    # Calculate recall
    recall = TP / (TP + FN + 1e-8)

    # Average over the batch dimension
    avg_recall = torch.mean(recall)

    return avg_recall


def specificity_time_series(pred, gt, threshold=0.5):
    # Convert to binary form using threshold
    pred_binary = (pred > threshold).float()
    gt_binary = (gt > threshold).float()

    # Compute True Negatives
    TN = torch.sum((1 - pred_binary) * (1 - gt_binary), dim=1)

    # Compute False Positives
    FP = torch.sum(pred_binary * (1 - gt_binary), dim=1)

    # Calculate specificity
    specificity = TN / (TN + FP + 1e-8)

    # Average over the batch dimension
    avg_specificity = torch.mean(specificity)

    return avg_specificity


def precision_time_series(pred, gt, threshold=0.5):
    # Convert to binary form using threshold
    pred_binary = (pred > threshold).float()
    gt_binary = (gt > threshold).float()

    # Compute True Positives
    TP = torch.sum(pred_binary * gt_binary, dim=1)

    # Compute False Positives
    FP = torch.sum(pred_binary * (1 - gt_binary), dim=1)

    # Avoid division by zero
    denom = TP + FP + 1e-8

    # Calculate precision
    precision = TP / denom

    # Average over the batch dimension
    avg_precision = torch.mean(precision)

    return avg_precision


def f2_score_time_series(pred, gt, threshold=0.5):
    p = precision_time_series(pred, gt, threshold)
    r = recall_time_series(pred, gt, threshold)

    # Calculate F2 score
    f2 = (5 * p * r) / (4 * p + r + 1e-8)

    return f2


def dice_coefficient_time_series(pred, gt, threshold=0.5):
    # Convert to binary form using threshold
    pred_binary = (pred > threshold).float()
    gt_binary = (gt > threshold).float()

    # Compute True Positives
    TP = torch.sum(pred_binary * gt_binary, dim=1)

    # Compute False Positives
    FP = torch.sum(pred_binary * (1 - gt_binary), dim=1)

    # Compute False Negatives
    FN = torch.sum((1 - pred_binary) * gt_binary, dim=1)

    # Calculate Dice coefficient
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)

    # Handle the case where both pred and gt are all zeros
    both_zero_mask = (torch.sum(gt_binary, dim=1) == 0) & (
        torch.sum(pred_binary, dim=1) == 0
    )
    dice[both_zero_mask] = 1.0

    # Average over the batch dimension
    avg_dice = torch.mean(dice, dim=0)

    return avg_dice


def euclidean_distance_time_series(pred, gt):
    # Compute the squared difference
    diff_squared = (pred - gt) ** 2

    # Sum over the time dimension and take the square root
    distances = torch.sqrt(torch.sum(diff_squared, dim=1))

    # Average over the batch dimension
    avg_distance = torch.mean(distances)

    return avg_distance


def compute_alls_in_segmentation(pred, gt, threshold=0.5):
    # Dice F1 score
    Dice_score = dice_coefficient_time_series(pred, gt)
    # F2
    F2_score = f2_score_time_series(
        pred, gt, threshold=threshold
    )  # could be nan
    # iou:
    iou = iou_time_series(pred, gt)

    # recall:
    recall = recall_time_series(
        pred, gt, threshold=threshold
    )  # could be nan again

    # specificity
    specificity = specificity_time_series(
        pred, gt, threshold=threshold
    )  # nan sometimes

    # precision
    precision = precision_time_series(pred, gt, threshold=threshold)

    euclidean = euclidean_distance_time_series(pred, gt)

    loss = compute_loss(pred, gt)
    return (
        torch.tensor(
            [
                Dice_score,
                F2_score,
                iou,
                recall,
                specificity,
                precision,
                euclidean,
                loss,
            ]
        )
        .detach()
        .cpu()
        .numpy()
    )


# -------------------------------------------------------------------------------------------------------------------------------------
def compute_loss(pred, gt):
    # pred shape is [batch_size, 1, length], gt shape is [batch_size, 1, length]
    return smooth_l1_loss(pred, gt).sum().item()


# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# from sklearn.metrics.pairwise import paired_euclidean_distances
# from sklearn.metrics import paired_distances


def relative_change_point_distance(cps_true, cps_pred, ts_len, penalty=1):
    """
    Calculates the relative CP distance between ground truth and predicted change points.
    Parameters
    Inspired by G.J.J. van den Burg (https://github.com/alan-turing-institute/TCPDBench)
    We added a penalty term on rcps to account for missing or additional change points.
    -----------
    :param cps_true: an array of true change point positions
    :param cps_pred: an array of predicted change point positions
    :param ts_len: the length of the associated time series
    :param penalty: the penalty for each additional or missing change point
    :return: relative distance between cps_true and cps_pred considering ts_len

    lower better
    """
    cps_true = np.array(cps_true)
    cps_pred = np.array(cps_pred)
    # Calculate distances from each predicted CP to the nearest true CP
    pred_to_true_distances = np.array(
        [
            np.min(np.abs(cp_pred - cps_true)) if cps_true.size > 0 else np.inf
            for cp_pred in cps_pred
        ]
    )

    # Calculate distances from each true CP to the nearest predicted CP
    true_to_pred_distances = np.array(
        [
            np.min(np.abs(cp_true - cps_pred)) if cps_pred.size > 0 else np.inf
            for cp_true in cps_true
        ]
    )

    # Combine the distances
    all_distances = np.concatenate(
        (pred_to_true_distances, true_to_pred_distances)
    )

    # Calculate penalties for additional or missing CPs
    num_missing_or_additional = np.abs(len(cps_pred) - len(cps_true))
    penalty_term = num_missing_or_additional * penalty * ts_len

    # Combine distances and penalties
    total_difference = np.sum(all_distances) + penalty_term

    # Normalize by the maximum possible distance (number of CPs times ts_len)
    max_possible_distance = (len(cps_true) + len(cps_pred)) * ts_len
    relative_distance = total_difference / max_possible_distance

    return np.round(relative_distance, 6)


# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------


def _overlap(A, B):
    """
    Return the overlap (i.e. Jaccard index) of two sets, adjusted for partial coverage.

    adjusted for partial coverage:
    i.e.
    gt: {1000, 1350} pred: {1000, 1300} would be 1 originally
    but now it is 0.95 (1 - (50/350))
    """
    intersection = len(A.intersection(B))
    union = len(A.union(B))
    # If one segment is completely within another but their lengths differ,
    # penalize the overlap by the difference in lengths divided by the length of the union.
    length_difference = abs(len(A) - len(B))
    return (intersection - length_difference) / union if union > 0 else 0


def _partition_from_intervals(intervals):
    """
    Create a partition of the observation sequence based on the provided intervals.
    Each set in the partition represents a segment from start to end.
    Author: Hanchen David Wang
    Inspired by G.J.J. van den Burg (https://github.com/alan-turing-institute/TCPDBench)

    """
    partition = []
    for start, end in intervals:
        partition.append(
            set(range(start, end + 1))
        )  # +1 because range end is exclusive
    return partition


def _covering_single(Sprime, S):
    """
    Compute the covering of a segmentation S by a segmentation Sprime.
    Each element of Sprime and S is a set representing a segment.
    Author: Hanchen David Wang
    Inspired by G.J.J. van den Burg (https://github.com/alan-turing-institute/TCPDBench)

    """
    T = sum(len(s) for s in Sprime)
    C = 0
    if T > 0:  # Check if Sprime is not empty
        for R in S:
            overlaps = (_overlap(R, Rprime) for Rprime in Sprime)
            C += len(R) * max(
                overlaps, default=0
            )  # Use default=0 to handle empty sequence
        C /= T
    return C


def covering_repetition(annotations, predictions, n_obs):
    """
    Compute the average segmentation covering against the human annotations.
    Author: Hanchen David Wang
    Inspired by G.J.J. van den Burg (https://github.com/alan-turing-institute/TCPDBench)
    """
    # Convert annotations and predictions into partitions
    A = _partition_from_intervals(annotations)
    P = _partition_from_intervals(predictions)
    # Calculate covering for the annotations
    C = _covering_single(P, A)
    return C


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
