from torch.utils.data import Dataset

class PhysiQ(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.data = self.load_data()
        
    def load_data(self):
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def load_data(self):
        pass