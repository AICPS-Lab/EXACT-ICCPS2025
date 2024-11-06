DEFAULT_INFO = "data is a dictionary with keys as subject and values as list of data files, label is a dictionary with keys as subject and values as list of labels, label_to_exer_rom is a list of tuples with each tuple as (exercise, rom)"

# PhysiQ.py
import os
from .QueryDataset import QueryDataset
import pandas as pd
import torch


class PhysiQ(QueryDataset):
    DATASET_NAME = "physiq"

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
        loocv=False,
    ):
        super(PhysiQ, self).__init__(
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
        # PhysiQ filenames: [S1]_[E1]_[left-right hand]_[variation (range of motion)]_[stability]_[repetition_id].csv
        parts = filename.rstrip('.csv').split('_')
        if len(parts) < 6:
            return None  # Filename does not match expected pattern
        subj = parts[0]
        exer = parts[1]
        rom = parts[3]  # Range of motion
        ind_label = (exer, rom)
        unique_identifier = (subj, exer, rom)
        return {
            'subject': subj,
            'ind_label': ind_label,
            'unique_identifier': unique_identifier,
        }

    def get_dataset_name(self):
        return self.DATASET_NAME

    def default_data_folders(self):
        # The PhysiQ dataset has multiple folders for each exercise
        folders = [
            "segment_sessions_one_repetition_data_E1",
            "segment_sessions_one_repetition_data_E2",
            "segment_sessions_one_repetition_data_E3",
            "segment_sessions_one_repetition_data_E4",
        ]
        return [os.path.join(self.root, self.DATASET_NAME, folder) for folder in folders]

    # Implement any dataset-specific methods if needed.
