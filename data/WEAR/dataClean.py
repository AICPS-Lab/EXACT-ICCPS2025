import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
activities = ['null', 'bench-dips', 'lunges', 'burpees', 'jogging (sidesteps)', 'sit-ups', 'jogging', 'stretching (shoulders)', 
              'push-ups', 'push-ups (complex)', 'jogging (skipping)', 'jogging (butt-kicks)', 'stretching (triceps)', 
              'stretching (hamstrings)', 'stretching (lunging)', 'jogging (rotating arms)', 'sit-ups (complex)', 'lunges (complex)',
              'stretching (lumbar rotation)']
def process_directory(folder):
    # Read all files in the folder
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    # Read all files in the folder
    datas = []
    labels = []
    mapping = {each: activities.index(each) for each in activities}
    print(mapping)
    for f in files:
        # Read the file
        df = pd.read_csv(os.path.join(folder, f))
        # Append the file to the data
        mask = df['label'].isin(['burpees', 'push-ups'])
        # check if each row is number or float:
        datas.append(df[mask].to_numpy()[:, 1:4])
        
        
        df['label'] = df['label'].map(mapping)
        df['label'] = df['label'].fillna(0)
        labels.append(df[mask].to_numpy()[:, -1])
        # mapping from the list using its index:
        
    datas = np.concatenate(datas)
    datas = datas.astype(np.float64)
    labels = np.concatenate(labels, dtype=np.float64)
    print(np.unique(labels), datas.dtype, labels.dtype)
        
    return datas, labels

# def process_data(folder):
#     # Read all files in the folder
#     datas, labels = process_directory(folder)
#     for i in zip(datas, labels):
#         print(i)


if __name__ == "__main__":
    i, l = process_directory('./datasets/WEAR/inertial')
    np.save('./datasets/WEAR/inertial.npy', {'data': i, 'labels': l})
    