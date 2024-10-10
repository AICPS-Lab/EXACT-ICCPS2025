from matplotlib import pyplot as plt
import numpy as np
import torch


def mean_iou_time_series(pred, gt, num_classes=7):
    """
    Calculate the Mean Intersection over Union (mIoU) for time series data.
    
    Parameters:
        pred (torch.Tensor): Predictions, assumed to be one-hot encoded.
        gt (torch.Tensor): Ground truth labels, assumed to be one-hot encoded.
        num_classes (int): Number of classes in the segmentation task.

    Returns:
        float: Mean IoU for the batch.
    """
    # Initialize variables to store IoU for each class
    iou_list = []

    # Loop through each class
    for cls in range(num_classes):
        pred_inds = pred == cls
        gt_inds = gt == cls

        # Calculate Intersection and Union
        # print(pred_inds.shape, gt_inds.shape)
        intersection = torch.sum(pred_inds & gt_inds, dim=1).float()
        union = torch.sum(pred_inds | gt_inds, dim=1).float()

        # Calculate IoU. Avoid division by zero by adding a small epsilon.
        iou = intersection / (union + 1e-8)

        # Append the mean IoU for this class
        iou_list.append(torch.mean(iou))

    # Calculate the mean IoU across all classes
    mean_iou = torch.mean(torch.stack(iou_list))

    return mean_iou.item()


def mean_iou(preds, labels, num_classes):
    # Flatten the predictions and labels, this comes from the argmax of the one-hot vectors
    preds = preds.view(-1)
    labels = labels.view(-1)

    # Create confusion matrix
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for i in range(num_classes):
        for j in range(num_classes): 
            confusion_matrix[i, j] = torch.sum((preds == i) & (labels == j))

    # Calculate IoU for each class
    ious = []
    for i in range(num_classes):
        true_positive = confusion_matrix[i, i]
        false_positive = confusion_matrix[i, :].sum() - true_positive
        false_negative = confusion_matrix[:, i].sum() - true_positive

        # Avoid division by zero
        union = true_positive + false_positive + false_negative
        if union == 0:
            ious.append(float('nan'))  # No predictions and no labels for this class
        else:
            ious.append(true_positive.float() / union.float())

    # Calculate mean IoU
    ious = torch.tensor(ious)
    mean_iou = torch.nanmean(ious)  # Mean over all classes, ignoring NaNs
    return mean_iou

def find_majority_label(arr):
    results = []
    for i in range(arr.shape[0]):
        # Process each row/sample individually
        row = arr[i]
        
        # Calculate differences along the sequence
        diffs = torch.diff(row)
        
        # Find indices where the value changes
        change_indices = torch.where(diffs != 0)[0] + 1
        
        # Select the unique elements, including the first element of the sequence
        unique_elements = torch.cat((row[0:1], row[change_indices]))
        
        results.append(unique_elements.cpu().tolist())  # Assuming you want the final result as a NumPy array
    
    return results

# Define the visualization function with subplots, including the label subplot
def visualize_softmax(pred, label_data, data):
    time = np.arange(pred.shape[1])
    labels = np.arange(pred.shape[0])
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Original data plot
    axs[0].plot(time, data, label='Original Data')
    axs[0].set_title('Original Input Data')
    axs[0].set_ylabel('Value')
    axs[0].legend(loc='upper left')

    # Labels plot
    colors = plt.cm.viridis(np.linspace(0, 1, pred.shape[0]))  # Get a colormap to use for the label lines
    for i in range(len(labels)):
        print(time[label_data == i])
        axs[1].fill_between(time, 0, 1, label=f'Label {i}', color=colors[i], where=label_data == i, alpha=0.7)
    axs[1].set_title('Labels')
    axs[1].set_ylabel('Label Active')
    # axs[1].set_ylim(-0.1, 1.1)  # Set y limits to make the lines more distinct
    axs[1].legend(loc='upper left')

    # Softmax probabilities plot
    start = np.zeros(pred.shape[1])  # Initialize the starting boundary as zeros
    for i in range(pred.shape[0]):
        axs[2].fill_between(time, start, start + pred[i, :], label=labels[i], alpha=0.7, color=colors[i])
        start += pred[i, :]  # Update the starting boundary

    axs[2].plot(time, start, color='black', lw=2, label='Total Confidence')
    axs[2].set_title('Softmax Probabilities with Stacked Areas')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Confidence Score')
    axs[2].set_ylim(0, 1.1)
    axs[2].legend(loc='upper left')

    plt.tight_layout()
    # plt.show()
    # plt.savefig
    plt.savefig('softmax_plot.png')



def eval_dense_label_to_classification(preds, labels):
    """
    Evaluate dense predictions to classification labels.
    
    Parameters:
        preds (torch.Tensor): Dense predictions, assumed to be one-hot encoded.
        labels (torch.Tensor): Ground truth labels

    Returns:
        torch.Tensor: Classification labels.
    """
    
    # if the preds is like (N, 300, C) where N is batch, C is # of classes
    softmaxed = torch.nn.functional.softmax(preds[0], dim=0)
    preds_labels = torch.argmax(preds, dim=1)
    preds_labels = find_majority_label(preds_labels)
    targs_labels = find_majority_label(labels)
    return
    