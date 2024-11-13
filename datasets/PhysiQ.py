DEFAULT_INFO = "data is a dictionary with keys as subject and values as list of data files, label is a dictionary with keys as subject and values as list of labels, label_to_exer_rom is a list of tuples with each tuple as (exercise, rom)"

# PhysiQ.py
import os

import pandas as pd
import torch

from .QueryDataset import QueryDataset


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
        test_subject: int = None,
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
        )

    def parse_filename(self, filename):
        # PhysiQ filenames: [S1]_[E1]_[left-right hand]_[variation (range of motion)]_[stability]_[repetition_id].csv
        parts = filename.rstrip(".csv").split("_")
        if len(parts) < 6:
            return None  # Filename does not match expected pattern
        subj = parts[0]
        exer = parts[1]
        rom = parts[3]  # Range of motion
        ind_label = (exer, rom)
        unique_identifier = (subj, exer, rom)
        return {
            "subject": subj,
            "ind_label": ind_label,
            "unique_identifier": unique_identifier,
        }
        
    def get_variation(self):
        return 3 # based on the file name, the indices of split('_')

    def get_subject(self):
        # this is not the subject of the file name but the subject for splitting or shuffling the data:
        # PhysiQ filenames: [S1]_[E1]_[left-right hand]_[variation (range of motion)]_[stability]_[repetition_id].csv
        # returning the index of the subject in the filename
        return [0, 1]

    def get_ind_label(self):
        # PhysiQ filenames: [S1]_[E1]_[left-right hand]_[variation (range of motion)]_[stability]_[repetition_id].csv
        # returning the index of the label in the filename
        return [1, 2] # from the tuple: (subj, exer, rom)

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
        return [
            os.path.join(self.root, self.DATASET_NAME, folder)
            for folder in folders
        ]

    def data_correspondence(self):
        # 6 exercises, 1 hands, 5 variations for 0, 2 and 3 variation for 1, 3
        dic = {
            0: [0, 1, 2, 3, 4],
            1: [5, 6, 7],
            2: [8, 9, 10, 11, 12],
            3: [
                13,
                14,
                15,
            ],
        }
        if self.args.add_side_noise:
            for k, v in dic.items():
                for i in range(len(v)):
                    v[i] += 1
        return dic

    # Implement any dataset-specific methods if needed.
