import os
from few_shot_models import time_FewShotSeg
from methods import Segmenter
from matplotlib import pyplot as plt
import torch
from utilities import printc
from utils_loader import get_dataloaders
from torch.optim.lr_scheduler import MultiStepLR

def main():
    config = {
        'allowed_label': [1, 2,3,4],
        'n_way': 3,
        'n_shot': 4,
        'batch_size': 2,
        
        'n_query': 2,
        'n_tasks_per_epoch': 500,
        'align': True,
        'fsl': True,
        'epochs':10
    }
    train_loader, test_loader = get_dataloaders(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    segmenter = Segmenter(embed_dims=256, num_classes=5).float().to(device)
    segmenter.load_state_dict(torch.load('./saved_model/best_transformer_oppo.pth'))
    model = time_FewShotSeg(segmenter, device=device, cfg=config).float().to(device)
    
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
            fg_mask = support_mask
            bg_mask = torch.where(fg_mask == 0, 1, 0).to(device)
            optimizer.zero_grad()
            query_pred, align_loss = model(support_images, fg_mask, bg_mask, query_images)
            query_mask = query_mask.view(-1, *query_mask.shape[-1:]).long()
            query_loss = criterion(query_pred, query_mask)
            loss = query_loss + align_loss * 1
            loss.backward()
            optimizer.step()
            scheduler.step()
            if i_iter % 50 == 0:
                query_indx = torch.argmax(query_pred, dim=1)
                printc(f'Iter {i_iter}, Loss: {loss.item()}, query_loss: {query_loss.item()}, align_loss: {align_loss.item()}')

        plt.plot(query_indx[0].detach().cpu().numpy(), color='r')
        plt.plot(query_mask[0].detach().cpu().numpy(), color='b')
        # plt.show()
        # get the num of directory in 'results':
        
        plt.savefig(f'./results/{i}/{i_epoch}.png')
        plt.close()
        # plt.savefig(f'./results/{i}.png')
    
    
    
    
if __name__ == "__main__":
    main()
    
        