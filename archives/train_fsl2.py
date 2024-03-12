import copy
import os
from few_shot_models import time_FewShotSeg
from methods import Segmenter
from matplotlib import pyplot as plt
import torch
from train_dense_labeling import get_model
from utilities import printc
from utils_loader import get_dataloaders
from torch.optim.lr_scheduler import MultiStepLR

from utils_metrics import mean_iou

def main(config):
    
    train_loader, test_loader = get_dataloaders(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config).float().to(device)
    
    # train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[30, 100, 200], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    model.train()
    i = len(os.listdir('./results'))
    # create a new directory in results:
    if not os.path.exists(f'./results/{i}'):
        os.mkdir(f'./results/{i}')
    for i_epoch in range(config['epochs']):
        for i_iter, sample_batch in enumerate(train_loader):
            support_images, support_mask, support_labels, query_images, query_mask, query_labels, classes = sample_batch
            
            support_images = support_images.float().to(device)
            support_mask = support_mask.float().to(device)
            query_images = query_images.float().to(device)
            query_mask = query_mask.float().to(device)
            
            optimizer.zero_grad()
            support_images = support_images.view(-1, support_images.shape[-2], support_images.shape[-1])
            support_mask = support_mask.view(-1, support_mask.shape[-1])
            support_pred = model(support_images)
            support_pred = support_pred.permute(0, 2, 1)
            loss = criterion(support_pred, support_mask.long())
   
            loss.backward()
            optimizer.step()
            query_images = query_images.view(-1, query_images.shape[-2], query_images.shape[-1])
            query_mask = query_mask.view(-1, query_mask.shape[-1])
            miou = mean_iou(model.forward_pred(query_images), query_mask.long(), num_classes=config['n_way'])
            # scheduler.step()
            
            outputs = model(query_images)
            outputs = outputs.permute(0, 2, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            accuracy = (preds == query_mask.long()).sum().item() / (query_mask.shape[0] * query_mask.shape[1])
            if i_iter % 50 == 0:
                printc(f"Iter {i_iter}, Train Loss: {loss.item()}, Train Query mIoU: {miou.item()}, accuracy: {accuracy}")
        
        # test the model
        temp_model = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-2) # inference for FSL
        # model.eval() # it is eval mode but requires inductive training 
        mious = []
        accus = []
        for i_iter, sample_batch in enumerate(test_loader):
            
            support_images, support_mask, support_labels, query_images, query_mask, query_labels, classes = sample_batch
            support_images = support_images.float().to(device)
            support_mask = support_mask.float().to(device)
            query_images = query_images.float().to(device)
            query_mask = query_mask.float().to(device)
            for _ in range(5):
                optimizer.zero_grad()
                support_images = support_images.view(-1, support_images.shape[-2], support_images.shape[-1])
                support_mask = support_mask.view(-1, support_mask.shape[-1])
                support_pred = temp_model(support_images)
                support_pred = support_pred.permute(0, 2, 1)
                loss = criterion(support_pred, support_mask.long())
                loss.backward()
                optimizer.step()
            query_images = query_images.view(-1, query_images.shape[-2], query_images.shape[-1])
            query_mask = query_mask.view(-1, query_mask.shape[-1])
            miou = mean_iou(temp_model.forward_pred(query_images), query_mask.long(), num_classes=config['n_way'])
            outputs = temp_model(query_images)
            outputs = outputs.permute(0, 2, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            accuracy = (preds == query_mask.long()).sum().item() / (query_mask.shape[0] * query_mask.shape[1])
            mious.append(miou.item())
            accus.append(accuracy)
        print(f"Test mIoU: {sum(mious) / len(mious)}, Test accuracy: {sum(accus) / len(accus)}")


        # plt.plot(query_indx[0].detach().cpu().numpy(), color='r')
        # plt.plot(query_mask[0].detach().cpu().numpy(), color='b')
        # plt.show()
        # get the num of directory in 'results':
        
        # plt.savefig(f'./results/{i}/{i_epoch}.png')
        # plt.close()
        # plt.savefig(f'./results/{i}.png')
    
    
    
    
if __name__ == "__main__":
    config = {
        'allowed_label': [1, 2,3,4],
        'n_way': 3,
        'n_shot': 4,
        'batch_size': 2,
        
        'n_query': 2,
        'n_tasks_per_epoch': 500,
        'fsl': True,
        'epochs':10,
        'dataset': 'opportunity',
        'model': 'crnn',
    }
    main(config)
    
        