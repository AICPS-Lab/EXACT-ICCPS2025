from torch.utils.data import Dataset

from utilities import sliding_windows


class QueryDataset(Dataset):
    ## time series data, spliting dataset in support and query set
    ## always 2-way N-shot 2 ways are 0 and 1, 0 is the all the other classes, 1 is the class of interest
    ## support set is the set of examples used to learn the model
    ## query set is the set of examples used to test the model
    ## N-shot is the number of examples used to learn the model
    ## in here, N-shot represents from the same 'class' or exercises
    def __init__(
        self,
        root="data",
        split="train",
        window_size=300,
        window_step=50,
        bg_fg=None,
        args=None,
    ):
        self.root = root
        # if bg_fg is not None:
        #     if N_way != 2:
        #         Warning("N_way is set to 2, because bg_fg is set to True")
        #     N_way = 2
        # self.N_way = N_way
        assert split in ["train", "test"], f"Invalid split: {split}"
        self.bg_fg = bg_fg
        self.split = split
        self.window_size = window_size
        self.window_step = window_step
        self.args = args

        self.sw = sliding_windows(window_size, window_step)
        if not self.if_npy_exists(split):
            self._process_data()

        self.data = self.load_data(split)
        self.data, self.label, self.res_exer_label = self.concanetate_data()

    def concanetate_data(self):
        raise NotImplementedError

    def _process_data(self):
        raise NotImplementedError

    def load_data(self, split):
        raise NotImplementedError

    def if_npy_exists(self, split):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx], self.res_exer_label[idx]
