import copy
import os
from prototypical   import PrototypicalNetworks
from methods import Segmenter
from matplotlib import pyplot as plt
import torch
from train import get_model
from utilities import printc
from utils_loader import get_dataloaders
from torch.optim.lr_scheduler import MultiStepLR

from utils_metrics import mean_iou

def main(config):
    train_loader, test_loader = get_dataloaders(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config).float().to(device)
    model.load_state_dict(torch.load(f'./saved_model/opportunity_{config["model"]}.pth', map_location=device))
    
    # check the data:
    for i_iter, sample_batch in enumerate(train_loader):
        support_images, support_mask, support_labels, query_images, query_mask, query_labels, classes = sample_batch
        # print(support_images.shape, support_mask.shape, support_labels.shape, query_images.shape, query_mask.shape, query_labels.shape, classes)
        support_images_res = support_images.reshape(-1, support_images.shape[-2], support_images.shape[-1])
        support_mask_res = support_mask.reshape(-1, support_mask.shape[-1])
        print(support_images_res.shape, support_mask_res.shape)
        
        res = model(support_images_res)
        
        break
    

if __name__ == '__main__':
    config = {
        'batch_size': 128,
        'epochs':200,
        'fsl': True,
        'model': 'unet',
        'seed': 73054772,
        'dataset': 'physiq',
        'allowed_label': [0, 1, 2,3,4],
        'n_way': 3,
        'n_shot': 4,
        'batch_size': 2,
        'n_query': 2,
        'n_tasks_per_epoch': 500,
        
    }
    main(config)