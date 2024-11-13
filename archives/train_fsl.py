import copy
import os
from prototypical   import PrototypicalNetworks
from methods import Segmenter
from matplotlib import pyplot as plt
import torch
from train_dense_labeling import get_model
from utilities import printc, seed
from archives.utils_loader import get_dataloaders
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from utils_metrics import mean_iou

def main(config):
    seed(config['seed'])
    train_loader, test_loader = get_dataloaders(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config).float().to(device)
    model.load_state_dict(torch.load(f'./saved_model/opportunity_{config["model"]}.pth', map_location=device))
    model.final_conv = torch.nn.Identity()
    # check the data:
    for i_iter, sample_batch in enumerate(train_loader):
        support_images, support_mask, support_labels, query_images, query_mask, query_labels, real_classes_id = sample_batch
        # print(support_images.shape, support_mask.shape, support_labels.shape, query_images.shape, query_mask.shape, query_labels.shape, classes)
        support_images_res = support_images.reshape(-1, support_images.shape[-2], support_images.shape[-1]).float().to(device)
        support_mask_res = support_mask.reshape(-1, support_mask.shape[-1]).float().to(device)
        
        query_images_res = query_images.reshape(-1, query_images.shape[-2], query_images.shape[-1]).float().to(device)
        query_mask_res = query_mask.reshape(-1, query_mask.shape[-1]).float().to(device)
        
        res = model(support_images_res)
        res = res.reshape(*support_images.shape[0:4], -1)
        
        z_query = model(query_images_res)
        # z_query = z_query.reshape(*query_images.shape[0:4], -1)
        
        class_labels_id = torch.unique(support_mask) # [0,1, 2, 3]
        class_prototypes = []
        # print(res[support_mask==0].shape)
        class_prototypes = [res[support_mask==0].mean(0)]
        for i in class_labels_id[1::]:
            # support labels only has encoded labels [0,1,2], but class labels from mask also have bg labels 0
            # therefore, we need to subtract 1 from the class_labels_id:
            # res_i = res[support_labels==(i-1)] 
            # res_label_i = support_mask[support_labels==(i-1)]
            # res_i_reshape = res_i.reshape(-1, *res_i.shape[1::]) # n_support x batch_size, 300, 7
            # res_label_i_reshape =  res_label_i.reshape(-1, *res_label_i.shape[1::]) # n_support x batch_size, 300
            # all_prototypes = []
            # for each_sample, each_label in zip(res_i_reshape,res_label_i_reshape):
            #     extracted = each_sample[each_label == i] # e.g.: 300,64 -> 234,64
            #     # 234, 64 -> 64: average:
            #     each_prototye = torch.mean(extracted, dim=0)
            #     all_prototypes.append(each_prototye)
            # all_prototypes = torch.stack(all_prototypes)
            # all_prototypes = torch.mean(all_prototypes, dim=0)
            # class_prototypes.append(all_prototypes)
            class_prototypes.append(res[support_labels==(i-1)][support_mask[support_labels==(i-1)]==i].mean(0))
        class_prototypes = torch.stack(class_prototypes)
        class_prototypes = class_prototypes.unsqueeze(0).unsqueeze(1).expand(*z_query.shape[0:2], -1, -1)  # Shape becomes [6, 300, 3, 64]
        z_query = z_query.unsqueeze(2)  # 
        
        # calculate the distance:
        res = F.cosine_similarity(z_query, class_prototypes, dim=-1)
        predicted = torch.argmax(res,dim=-1)
        print((predicted == query_mask_res).sum().item() / (query_mask_res.shape[0] * query_mask_res.shape[1]))
        # break
    

if __name__ == '__main__':
    config = {
        # 'batch_size': 128,
        'epochs':200,
        'fsl': True,
        'model': 'unet',
        'seed': 73054772,
        'dataset':{ 'name': 'physiq_e3', 
                   'num_classes': 2},
        'allowed_label': [0, 1, 2, 3, 4],
        'n_way': 3,
        'n_shot': 4,
        'batch_size': 1,
        'n_query': 2,
        'n_tasks_per_epoch': 10,
        
    }
    main(config)