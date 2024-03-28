import math
import os
from methods import Segmenter, TransformerModel, LSTM, CRNN, UNet,CCRNN, UNet2, PatchTST
from matplotlib import pyplot as plt
import torch
from methods.unetc import UNetc
from methods.unetr import UNetrt
from utilities import printc, seed
from utils_loader import get_dataloaders, test_idea_dataloader_ABC_to_BCA, test_idea_dataloader_long_A_B_to_AB
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils_metrics import eval_dense_label_to_classification, mean_iou, visualize_softmax
from torch.nn import functional as F

def get_model(config):
    if config['model'].lower() == 'crnn':
        return CRNN(in_channels=6, embed_dims=256, num_classes=config['dataset']['num_classes'])
    elif config['model'].lower() == 'lstm':
        return LSTM(num_classes=config['dataset']['num_classes'], in_channels=6, embed_dims=64)
    elif config['model'].lower() == 'unet':
        return UNet(in_channels=6, out_channels=config['dataset']['num_classes'])
    elif config['model'].lower() == 'ccrnn':
        return CCRNN(in_channels=6, embed_dims=256, num_classes=config['dataset']['num_classes'])
    elif config['model'].lower() == 'transformer':
        return TransformerModel(in_channels=6,embed_dims=256, num_classes=config['dataset']['num_classes'])
    elif config['model'].lower() == 'segmenter':
        return Segmenter(in_channels=6, embed_dims=128, num_classes=config['dataset']['num_classes'])
    elif config['model'].lower() == 'unet2':
        return UNet2(in_channels=6, out_channels=config['dataset']['num_classes'])
    elif config['model'].lower() == 'patchtst':
        return PatchTST()
    elif config['model'].lower() == 'unetrt':
        return UNetrt(in_channels=6, out_channels=config['dataset']['num_classes'])
    elif config['model'].lower() == 'unetc':
        return UNetc(in_channels=6, out_channels=config['dataset']['num_classes'])
    else:
        raise NotImplementedError

def main(config):
    
    seed(config['seed'])
    # train_loader, val_loader, test_loader = test_idea_dataloader_ABC_to_BCA(config, dense_label=True)# get_dataloaders(config)
    train_loader, val_loader, test_loader = test_idea_dataloader_long_A_B_to_AB(config, dense_label=True, interpolateA=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config).float().to(device)
    # model = UNet(in_channels=6, out_channels=5).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # print number of trainable parameters:
    printc('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = math.inf
    counter_i = 0
    pbar = tqdm(range(config['epochs']), postfix={'loss': 0.0, 'miou': 0.0})
    for epoch in pbar:
        model.train()
        for images, labels in train_loader:
            b, h, c = images.shape
            # Forward pass
            images = images.float().to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = outputs.permute(0, 2, 1)
            # probs = F.softmax(outputs, dim=1)
            loss= criterion(outputs, labels.long())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., norm_type=2)
            optimizer.step()
            # scheduler.step(loss)
            # train loss and accuracy check:
            # if counter_i % 1000 == 0 and counter_i != 0:
            #     print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, config['epochs'], loss.item()))
        val_losses = []
        mean_ious = []
        model.eval()
        for images, labels in val_loader:
            images = images.float().to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, labels.long())
            val_losses.append(loss.item())
            mean_ious.append(mean_iou(model.forward_pred(images), labels.long(),num_classes=config['dataset']['num_classes']))
        val_loss = sum(val_losses) / len(val_losses)
        miou = sum(mean_ious) / len(mean_ious)
        # print(f'Epoch [{epoch+1}/{config["epochs"]}], Val Loss: {val_loss}, Mean IoU: {miou}')
        pbar.set_postfix({'loss': val_loss, 'miou': miou.item()})
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'./saved_model/opportunity_{config["model"]}.pth')
        # visualize the predictions:
        # plt.plot(images[0, :].cpu().detach().numpy(), color='black')
        # plt.plot(model.forward_pred(images)[0, :].cpu().detach().numpy(), label='Prediction')
        # plt.plot(labels[0, :].cpu().detach().numpy(), label='Ground Truth')
        # plt.legend()
        # plt.show()
        counter_i += 1
    # Test the model
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        m_ious = []
        for images, labels in test_loader:
            images = images.float().to(device)
            labels = labels.to(device)
            outputs = model(images)
            outputs = outputs.permute(0, 2, 1) # B, C, T
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0) * labels.size(1)
            correct += (predicted == labels).sum().item()
            m_ious.append(mean_iou(model.forward_pred(images), labels.long(), num_classes=config['dataset']['num_classes']))
            
            softmaxed = F.softmax(outputs, dim=1)
            for i in range(images.shape[0]):
            # i = 0
                visualize_softmax(softmaxed[i].cpu().detach().numpy(), labels[i].cpu().detach().numpy(), images[i].cpu().detach().numpy())

            eval_dense_label_to_classification(outputs, labels)
            
        printc('Test Accuracy of the model {} on the test images: {} %, Mean IoU: {}'.format(config['model'].upper(), 
                                                                                            100 * correct / total, 
                                                                                            sum(m_ious) / len(m_ious)))

if __name__ == "__main__":
    config = {
        'batch_size': 128,
        'epochs':100,
        'fsl': False,
        'model': 'unet',
        'seed': 73054772,
        'dataset': {
            'name': 'physiq_e2',
            'num_classes': 2
        },
        
        'window_size': 300,
        'step_size': 150,
        'cross_subject': {
            'enabled': False,
            'n_groups': 1 # number of test group (subject)
        }
    }
    main(config)