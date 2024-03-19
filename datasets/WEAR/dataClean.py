import os
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
        datas.append(df[mask].to_numpy()[1:4])
        # extract certain rows given its label exercise:

        
        df['label'] = df['label'].map(mapping)
        df['label'] = df['label'].fillna(0)
        labels.append(df[mask].to_numpy()[-1])
        print(df[mask].head())
        # mapping from the list using its index:
        
        
        
    return datas, labels

# def process_data(folder):
#     # Read all files in the folder
#     datas, labels = process_directory(folder)
#     for i in zip(datas, labels):
#         print(i)

process_directory('./datasets/WEAR/inertial')

if __name__ == "__main__":
    i, l = process_directory('./datasets/WEAR/inertial')
    np.save('./datasets/WEAR/inertial.npy', {'data': i, 'labels': l})
    