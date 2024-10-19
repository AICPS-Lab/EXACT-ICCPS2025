from matplotlib import pyplot as plt
import numpy as np
import torch
from    torch.nn import functional as F
from torch import nn
def mse_loss(y_pred, y_true):
    return torch.sum((y_pred - y_true) ** 2)
def smooth_l1_loss(y_pred, y_true):
    abs_diff = torch.abs(y_pred - y_true)
    loss = torch.where(abs_diff < 1, 0.5 * abs_diff ** 2, abs_diff - 0.5)
    return loss.mean()


from torch import nn

class AdjustSmoothL1Loss(nn.Module):
    # https://github.com/chengyangfu/retinamask/blob/master/maskrcnn_benchmark/layers/adjust_smooth_l1_loss.py
    def __init__(self, num_features, momentum=0.1, beta=1. /9):
        # num_features: it is the output feature size (i.e. main_length, not num of channels)
        super(AdjustSmoothL1Loss, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.beta = beta
        self.register_buffer(
            'running_mean', torch.empty(num_features).fill_(beta)
        )
        self.register_buffer('running_var', torch.zeros(num_features))

    def forward(self, inputs, target, size_average=True):

        n = torch.abs(inputs -target)
        with torch.no_grad():
            if torch.isnan(n.var(dim=0)).sum().item() == 0:
                self.running_mean = self.running_mean.to(n.device)
                self.running_mean *= (1 - self.momentum)
                self.running_mean += (self.momentum * n.mean(dim=0))
                self.running_var = self.running_var.to(n.device)
                self.running_var *= (1 - self.momentum)
                self.running_var += (self.momentum * n.var(dim=0))


        beta = (self.running_mean - self.running_var)
        beta = beta.clamp(max=self.beta, min=1e-3)

        beta = beta.view(-1, self.num_features).to(n.device)
        cond = n < beta.expand_as(n)
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
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
            self.sigma = self.momentum * self.sigma + (1.0 - self.momentum) * current_sigma2
        
        less_than_one = (abs_diff * abs_diff < self.sigma).float()

        loss = less_than_one * 0.5 * self.sigma * abs_diff * abs_diff + (1.0 - less_than_one) * (abs_diff - 0.5 / torch.sqrt(self.sigma))
        
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
    
    # Calculate IOU
    iou = intersection / union

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
    # plt.plot(pred[0].detach().cpu().numpy())
    # plt.plot(gt[0].detach().cpu().numpy())
    # plt.show()
    # print('dice', dice)
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
    F2_score = f2_score_time_series(pred, gt, threshold=threshold) # could be nan
    # iou:
    iou = iou_time_series(pred, gt)

    # recall:
    recall = recall_time_series(pred, gt, threshold=threshold) # could be nan again

    # specificity
    specificity = specificity_time_series(pred, gt, threshold=threshold) # nan sometimes

    # precision
    precision = precision_time_series(pred, gt, threshold=threshold)

    euclidean = euclidean_distance_time_series(pred, gt)

    loss = compute_loss(pred, gt)
    return torch.tensor([Dice_score, F2_score, iou, recall, specificity, precision, euclidean, loss]).detach().cpu().numpy()
# -------------------------------------------------------------------------------------------------------------------------------------
def compute_loss(pred, gt):
    # pred shape is [batch_size, 1, length], gt shape is [batch_size, 1, length]
    return smooth_l1_loss(pred, gt).sum().item()
# -------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# from sklearn.metrics.pairwise import paired_euclidean_distances
# from sklearn.metrics import paired_distances

def relative_change_point_distance(cps_true, cps_pred, ts_len, penalty=1):
    '''
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
    '''
    cps_true = np.array(cps_true)
    cps_pred = np.array(cps_pred)
    # Calculate distances from each predicted CP to the nearest true CP
    pred_to_true_distances = np.array([np.min(np.abs(cp_pred - cps_true)) if cps_true.size > 0 else np.inf for cp_pred in cps_pred])


    # Calculate distances from each true CP to the nearest predicted CP
    true_to_pred_distances = np.array([np.min(np.abs(cp_true - cps_pred)) if cps_pred.size > 0 else np.inf for cp_true in cps_true])

    
    # Combine the distances
    all_distances = np.concatenate((pred_to_true_distances, true_to_pred_distances))
    
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
    '''
    Return the overlap (i.e. Jaccard index) of two sets, adjusted for partial coverage.
    
    adjusted for partial coverage:
    i.e.
    gt: {1000, 1350} pred: {1000, 1300} would be 1 originally
    but now it is 0.95 (1 - (50/350))
    '''
    intersection = len(A.intersection(B))
    union = len(A.union(B))
    # If one segment is completely within another but their lengths differ, 
    # penalize the overlap by the difference in lengths divided by the length of the union.
    length_difference = abs(len(A) - len(B))
    return (intersection - length_difference) / union if union > 0 else 0

def _partition_from_intervals(intervals):
    '''
    Create a partition of the observation sequence based on the provided intervals.
    Each set in the partition represents a segment from start to end.
    Author: Hanchen David Wang
    Inspired by G.J.J. van den Burg (https://github.com/alan-turing-institute/TCPDBench)
    
    '''
    partition = []
    for start, end in intervals:
        partition.append(set(range(start, end + 1)))  # +1 because range end is exclusive
    return partition

def _covering_single(Sprime, S):
    '''
    Compute the covering of a segmentation S by a segmentation Sprime.
    Each element of Sprime and S is a set representing a segment.
    Author: Hanchen David Wang
    Inspired by G.J.J. van den Burg (https://github.com/alan-turing-institute/TCPDBench)
    
    '''
    T = sum(len(s) for s in Sprime)
    C = 0
    if T > 0:  # Check if Sprime is not empty
        for R in S:
            overlaps = (_overlap(R, Rprime) for Rprime in Sprime)
            C += len(R) * max(overlaps, default=0)  # Use default=0 to handle empty sequence
        C /= T
    return C

def covering_repetition(annotations, predictions, n_obs):
    '''
    Compute the average segmentation covering against the human annotations.
    Author: Hanchen David Wang
    Inspired by G.J.J. van den Burg (https://github.com/alan-turing-institute/TCPDBench)
    '''
    # Convert annotations and predictions into partitions
    A = _partition_from_intervals(annotations)
    P = _partition_from_intervals(predictions)
    # Calculate covering for the annotations
    C = _covering_single(P, A)
    return C


if __name__ == "__main__":
    annotations = np.array([10, 20, 40, 50])
    predictions = np.array([10, 20, 40, 55])
    # print(relative_change_point_distance(annotations, predictions, 3000))
    
    # print(f_measure({1: [10, 20], 2: [30, 40], 3: [50, 55], 4: [100, 120]}, [10, 11]))
    
    print(covering_repetition([(150, 400), (700, 900), (1100, 1350)], [(150, 400), (700, 900), (1100, 1300)], 1500))
    
    # [(150, 400), (700, 900), (1100, 1350)]
# -------------------------------------------------------------------------------------------------------------------------------------