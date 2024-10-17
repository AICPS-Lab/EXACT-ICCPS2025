from torch.utils.data import Dataset
from .QueryDataset import QueryDataset
import os

class PhysiQ(QueryDataset):
    def __init__(self, root, N_way=2, split="train"):
        super(PhysiQ, self).__init__(root, N_way, split)

    def load_data(self):
        folders = ['segment_session_one_repetition_data_E1',
                   'segment_session_one_repetition_data_E2',
                   'segment_session_one_repetition_data_E3',
                   'segment_session_one_repetition_data_E4',]
        data_folder = os.path.join(self.root, 'physiq')
        # physiq dataset 4 exercises and each contains variation (rom), each variation will be one class
        
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



