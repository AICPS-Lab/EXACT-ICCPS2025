from torch.utils.data import Dataset


class QueryDataset(Dataset):
    ## time series data, spliting dataset in support and query set
    ## always 2-way N-shot 2 ways are 0 and 1, 0 is the all the other classes, 1 is the class of interest
    ## support set is the set of examples used to learn the model
    ## query set is the set of examples used to test the model
    ## N-shot is the number of examples used to learn the model
    ## in here, N-shot represents from the same 'class' or exercises
    def __init__(self, root="data", N_way=2, split="train"):
        self.root = root
        self.N_way = N_way
        self.split = split
        assert self.split in ["train", "test"]
        self.data = self.load_data()

    def load_data(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
