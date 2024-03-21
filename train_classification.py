import math
import os
from methods import Segmenter, TransformerModel, LSTM, CRNN, UNet,CCRNN, UNet2, PatchTST
from matplotlib import pyplot as plt
import torch
from utilities import printc, seed
from utils_loader import get_dataloaders, test_idea_dataloader_er_ir, test_idea_dataloader_burpee_pushup
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils_metrics import mean_iou
from torch.nn import functional as F
from train_dense_labeling import get_model
from methods.unet import UNet_encoder
from sklearn.metrics import f1_score 
def main(config):
    
    seed(config['seed'])
    train_loader, val_loader, test_loader = test_idea_dataloader_er_ir(config) #test_idea_dataloader_er_ir(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet_encoder(in_channels=6, out_channels=2, cnn_embed_dims=[64]).float().to(device)
    # model = UNet(in_channels=6, out_channels=5).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # print number of trainable parameters:
    printc('Number of trainable parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    criterion = torch.nn.CrossEntropyLoss()

    best_loss = math.inf
    counter_i = 0
    pbar = tqdm(range(config['epochs']), postfix={'loss': 0.0, 'acc': 0.0, 'f1': 0.0})
    for epoch in pbar:
        model.train()
        for images, labels in train_loader:
            b, h, c = images.shape
            # Forward pass
            images = images.float().to(device)
            labels = labels.to(device)
            outputs = model(images)
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
        accs = []
        f1s =[]
        model.eval()
        for images, labels in val_loader:
            images = images.float().to(device)
            labels = labels.to(device)
            outputs = model(images)
            # outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, labels.long())
            val_losses.append(loss.item())
            # accuracy against labels:
            predicted = model.forward_pred(images)
            # predicted == labels
            f1 = f1_score(predicted.cpu().detach().numpy(), labels.cpu().detach().numpy(), average='weighted')
            f1s.append(f1)
            acc = (predicted == labels).sum().item() / predicted.shape[0]
            accs.append(acc)
        f1s = sum(f1s) / len(f1s)
        val_loss = sum(val_losses) / len(val_losses)
        accs = sum(accs) / len(accs)
        # print(f'Epoch [{epoch+1}/{config["epochs"]}], Val Loss: {val_loss}, Mean IoU: {miou}')
        pbar.set_postfix({'loss': val_loss, 'acc': accs, 'f1': f1s})
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'./saved_model/opportunity_{config["model"]}_test_EXP2_B.pth')
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
        misclassified_c = 0
        num_transitions, correct_in_trasi = 0, 0

        num_middles, correct_in_middles = 0, 0


        for images, (labels, (transition, is_middle)) in test_loader:
            num_transitions += sum(transition)
            num_middles += sum(is_middle)
            images = images.float().to(device)
            labels = labels.to(device)
            outputs = model(images)
            # outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, labels.long())
            # accuracy against labels:
            predicted = model.forward_pred(images)
            # predicted == labels
            correct_in_trasi += (predicted[transition==1] == labels[transition==1]).sum().item()  
            correct_in_middles += (predicted[is_middle==1] == labels[is_middle==1]).sum().item()

            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            # get the misclassified images and save them in a folder for visualization with its ground truth:
            if misclassified_c < 20:
                misclassified = predicted != labels
                images_misclassified = images[misclassified]
                labels_misclassified = labels[misclassified]
                predicted_misclassified = predicted[misclassified]
                for i, (img, label, pred) in enumerate(zip(images_misclassified, labels_misclassified, predicted_misclassified)):
                    plt.plot(img.cpu().detach().numpy())
                    plt.title(f'Predicted: {pred.cpu().detach().numpy()} Ground Truth: {label.cpu().detach().numpy()}')
                    # plt.plot(pred.cpu().detach().numpy(), label='Prediction')
                    # plt.plot(label.cpu().detach().numpy(), label='Ground Truth')
                    # plt.legend()
                    plt.savefig(f'./misclassified/{misclassified_c}_{i}.png')
                    plt.close()
                misclassified_c += misclassified.sum().item()  

        printc('Percentage of middles:', num_middles / total)
        printc('Percentage of middles correctly classified {} out of middles, and {} out of all:'.format(correct_in_middles / num_middles, correct_in_middles / total))

        printc('Percentage of transitions:', num_transitions / total)
        printc('Percentage of transitions correctly classified {} out of transitions, and {} out of all:'.format(correct_in_trasi / num_transitions, correct_in_trasi / total))

        accs = correct / total
        
        print('Percentage of misclassified:', misclassified_c / total)
        print('Out of the all, how many middle incorrectly classified:', (num_middles - correct_in_middles) / total)
        print('Out of the all, how many transitions incorrectly classified:', (num_transitions - correct_in_trasi) / total)
        inacc = 1-accs
        print('out of the incorrectly predicted, how many are middle incorrectly classified:', (num_middles - correct_in_middles) / total / inacc)
        print('out of the incorrectly predicted, how many are transitions incorrectly classified', (num_transitions - correct_in_trasi) / total / inacc)

        print(f'Test Accuracy of the model on the test images: {accs}')
        # print(f'Mean IoU: {miou}'

if __name__ == "__main__":
    config = {
        'batch_size': 128,
        'epochs':200,
        'fsl': False,
        'model': 'unet',
        'seed': 73054772,
        'dataset': 'physiq',
        'window_size': 50,
        'step_size': 25,
        
    }
    main(config)