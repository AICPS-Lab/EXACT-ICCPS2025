
DEFAULT_INFO = "data is a dictionary with keys as subject and values as list of data files, label is a dictionary with keys as subject and values as list of labels, label_to_exer_rom is a list of tuples with each tuple as (exercise, rom)"

# PhysiQ.py
import os
from .QueryDataset import QueryDataset
import pandas as pd
import torch

class MMFIT(QueryDataset):
    DATASET_NAME = "mmfit"

    def __init__(
        self,
        root="data",
        split="train",
        window_size=300,
        window_step=50,
        bg_fg=None,
        args=None,
        transforms=None,
        test_subject: int =None,
    ):
        super(MMFIT, self).__init__(
            root=root,
            split=split,
            window_size=window_size,
            window_step=window_step,
            bg_fg=bg_fg,
            args=args,
            transforms=transforms,
            test_subject=test_subject,
        )

    def parse_filename(self, filename):
        # MMFIT filenames: [Subject]_[Exercise]_[left-right hand]_[repetition].csv
        parts = filename.rstrip('.csv').split('_')
        if len(parts) < 4:
            return None  # Filename does not match expected pattern
        subj = parts[0]
        exer = parts[1]
        hand = parts[2]
        ind_label = (exer, hand)
        unique_identifier = (subj, exer, hand)
        return {
            'subject': subj,
            'ind_label': ind_label,
            'unique_identifier': unique_identifier,
        }

    def get_subject(self):
        # this is not the subject of the file name but the subject for splitting or shuffling the data:
        # aka shuffled based on
        # MMFIT filenames: [Subject]_[Exercise]_[left-right hand]_[repetition].csv
        # returning the index of the subject in the filename
        return [0, 1]
    
    def get_ind_label(self):
        # returning the index of the label in the filename
        # this needs to match to the ind_label in the parse_filename
        return [1, 2]

    def get_dataset_name(self):
        return self.DATASET_NAME

    def default_data_folders(self):
        return [os.path.join(self.root, "mmfit", "mmfit_dataset")]
