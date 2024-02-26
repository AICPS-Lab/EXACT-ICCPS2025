from few_shot_models import time_FewShotSeg
from methods import Segmenter
from matplotlib import pyplot as plt
import torch
from utilities import printc
from utils_loader import get_dataloaders
# custom Dataset:

    

def main():
    config = {
        'n_way': 1,
        'n_shot': 4,
        'n_query': 2,
        'n_tasks_per_epoch': 500,
        'align': True,
    }
    train_loader, test_loader = get_dataloaders(config)
    device = torch.device('cpu')
    segmenter = Segmenter(embed_dims=64, num_classes=2).float().to(device)
    model = time_FewShotSeg(segmenter, device=device, cfg=config).float().to(device)
    
    # train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    model.train()
    for i_iter, sample_batch in enumerate(train_loader):
        support_images, support_mask, support_labels, query_images, query_mask, query_labels, classes = sample_batch
        fg_mask = support_mask
        bg_mask = [[torch.where(support_mask[i][j] ==1, 0, 1) for j in range(len(support_mask[0]))] for i in range(len(support_mask))]
        
        optimizer.zero_grad()
        query_pred, align_loss = model(support_images, fg_mask, bg_mask, query_images)

        query_loss = criterion(query_pred, torch.stack(query_mask).long())
        loss = query_loss + align_loss * 1
        loss.backward()
        optimizer.step()
        # scheduler.step()
        if i_iter % 50 == 0:
            query_indx = torch.argmax(query_pred, dim=1)
            printc(f'Iter {i_iter}, Loss: {loss.item()}, query_loss: {query_loss.item()}, align_loss: {align_loss.item()}')
            plt.plot(query_indx[0].detach().cpu().numpy(), color='r')
            plt.plot(query_mask[0].detach().cpu().numpy(), color='b')
            plt.show()
    
    
    
    
if __name__ == "__main__":
    main()
    
        