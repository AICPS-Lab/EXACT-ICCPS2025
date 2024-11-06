DEFAULT_INFO = "data is a dictionary with keys as subject and values as list of data files, label is a dictionary with keys as subject and values as list of labels, label_to_exer_hand is a list of tuples with each tuple as (exercise, hand)"
# SPAR.py
import os
from .QueryDataset import QueryDataset
import pandas as pd
import torch


class SPAR(QueryDataset):
    """SPAR dataset has [Subject]_[Exercise]_[left-right hand]_[repetition].csv files"""

    DATASET_NAME = "spar"

    def __init__(
        self,
        root="data",
        split="train",
        window_size=300,
        window_step=50,
        bg_fg=None,
        args=None,
        transforms=None,
        test_subject=None,
        loocv=False,
    ):
        super(SPAR, self).__init__(
            root=root,
            split=split,
            window_size=window_size,
            window_step=window_step,
            bg_fg=bg_fg,
            args=args,
            transforms=transforms,
            test_subject=test_subject,
            loocv=loocv,
        )

    def parse_filename(self, filename):
        # SPAR filenames: [Subject]_[Exercise]_[left-right hand]_[repetition].csv
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

    def get_dataset_name(self):
        return self.DATASET_NAME

    def default_data_folders(self):
        return [os.path.join(self.root, "spar", "spar_dataset")]

    
