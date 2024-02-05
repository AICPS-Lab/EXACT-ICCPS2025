from matplotlib import pyplot as plt
import torch
from config import Config
from dataset import default_splitter
from models import CNNModel, TransformerModel
from utilities import get_device
import torchvision.transforms as transforms

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, predicted, ground_truth):
        # Flatten the tensors
        # For predicted, we need to select the probabilities for the '1' class (last channel)
        predicted_flat = predicted.contiguous().view(-1)
        ground_truth_flat = ground_truth.contiguous().view(-1)

        # Compute intersection and union
        intersection = (predicted_flat * ground_truth_flat).sum()
        union = predicted_flat.sum() + ground_truth_flat.sum()

        # Compute Dice coefficient
        dice_coefficient = (2. * intersection + 1e-6) / (union + 1e-6)

        # Compute Dice loss
        dice_loss = 1 - dice_coefficient

        return dice_loss

class Manager:
    def __init__(self, config):
        self.config = config  # yaml file of config
        self.device = get_device()
        #  ntoken=6, ninp=512, nhead=8, nhid=2048, nlayers=6, dropout=0.5
        self.model = self._get_model()
        self.train_dl, self.test_dl = self._get_dataloader()     
        # Assume config contains other necessary components like optimizer, scheduler, etc.

    def __str__(self):
        pass

    def __del__(self):
        pass
    
    

    def train(self):
        self.model.train()
        train_loss = 0
        optimizer = self._get_optimizer()
        criterion = self._get_criterion()
        batch_size = self.config['train']['batch_size']
        epochs = self.config['train']['epochs']

        for epoch in range(epochs):
            for i, (data, label) in enumerate(self.train_dl):
                data = data.to(self.device)
                label = label.to(self.device).float()
                optimizer.zero_grad()
                output = self.model(data)
                # print(output.shape, label.shape)
                loss = criterion(output, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
                train_loss += loss.item()
                if i % 100 == 0 and i > 0:
                    cur_loss = train_loss / 100
                    print(f'Epoch {epoch}, iter {i}, loss {cur_loss}')
                    train_loss = 0
                    # check the accuracy of the model in this batch (classification):
                    preds = torch.argmax(output, dim=-1)
                    acc = (preds == label).float().mean()
                    print(f'Epoch {epoch}, iter {i}, acc {acc}')
                    
                # if i % 1000 == 0 and i > 0:
                #     fig = plt.figure()
                #     ax = fig.add_subplot(111)
                #     ax.plot(data[0, :, 0].cpu().detach().numpy())
                #     # output is sigmoid, use 0.5 as threshold to get binary prediction
                #     ax.plot((output[0, :].cpu().detach().numpy() > 0.5).astype(int), label='prediction')
                #     ax.plot(label[0, :].cpu().detach().numpy(), label='ground truth')
                #     plt.legend()
                #     plt.draw()
                #     plt.pause(2)
                #     plt.close()

    def _get_model(self):
        if self.config['model']['name'] == 'transformer':
            return TransformerModel(ntoken=6, 
                                      ninp=config['model']['dim_inp'],
                                      nhead=config['model']['num_heads'],
                                      nhid=config['model']['dim_model'],
                                      nlayers=config['model']['num_layers'],
                                      dropout=config['model']['dropout']).to(self.device)
        elif self.config['model']['name'] == 'cnn':
            return CNNModel(ntoken=6,
                            ninp=config['model']['dim_inp'],
                            nhead=config['model']['num_heads'],
                            nhid=config['model']['dim_model'],
                            nlayers=config['model']['num_layers'],
                            dropout=config['model']['dropout']).to(self.device)
        else:
            raise NotImplementedError
    
    def _get_optimizer(self):
        if self.config['train']['optimizer'] == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.config['train']['lr'])
        elif self.config['train']['optimizer'] == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.config['train']['lr'])
        else:
            raise NotImplementedError
        
    def _get_criterion(self):
        if self.config['train']['criterion'] == 'mse':
            return torch.nn.MSELoss()
        elif self.config['train']['criterion'] == 'cross_entropy':
            return torch.nn.CrossEntropyLoss()
        elif self.config['train']['criterion'] == 'dice':
            return DiceLoss()
        elif self.config['train']['criterion'] == 'bce':
            return torch.nn.BCELoss()
        elif self.config['train']['criterion'] == 'ce':
            return torch.nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

    def _get_dataloader(self):
        # get current directory:
        import os
        abs_path = os.path.abspath(__file__)
        datasets_folder = os.path.join(abs_path, 'datasets')
        if self.config['dataset'] == 'spar':
            folder_name = os.path.join(datasets_folder, 'spar')
        else:
            raise NotImplementedError
        mean = (0.5)
        std = (0.5)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        train_dataset, test_dataset = default_splitter(folder_name, config, split=False,)
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['train']['batch_size'], shuffle=True,)
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=self.config['train']['batch_size'], shuffle=True)
        
        
        
        
        
        


if __name__ == "__main__":
    config = Config("config.yaml")
    m = Manager(config)

    
    # m.train(train_dataset)