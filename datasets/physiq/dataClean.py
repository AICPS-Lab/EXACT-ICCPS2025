import os
import pandas as pd
import numpy as np


def parser_folder_permuting(folder):
    """
    Parse the folder and return the inputs and labels

    Args:
        folder (_type_): NA
    """
    files = []
    labels_mapping = []
    users_mapping = []
    for file in os.listdir(folder):
        if '.csv' not in file:
            continue
        files.append(file)
        if file.split('_')[3] not in labels_mapping:
            labels_mapping.append(file.split('_')[3])
        if file.split('_')[0] not in users_mapping:
            users_mapping.append(file.split('_')[0])
    
    # sort the labels_mapping:
    
    labels_mapping = sorted(labels_mapping)
    print(labels_mapping)
    # permute the order:
    np.random.shuffle(files)
    inputs = []
    labels = []
    users = []
    transition = []
    cur_value = 0
    for file in files:
        path_file = os.path.join(folder, file)
        df = pd.read_csv(path_file)
        inputs.append(df.iloc[:, 1:7].values)
        labels.append([labels_mapping.index(file.split('_')[3])] * df.shape[0])
        users.append(users_mapping.index(file.split('_')[0]))
        transition.append([cur_value] * df.shape[0])
        cur_value = 1 - cur_value
    
    # inputs = np.concatenate(inputs, axis=0)
    # labels = np.concatenate(labels, axis=0, dtype=int)
    # users = np.concatenate(users, axis=0)
    
    return inputs, labels, users, transition



            
if __name__ == "__main__":
    
    # i, l, u = parser_folder_permuting('./datasets/physiq/segment_sessions_one_repetition_data_E1')
    # np.save('./datasets/physiq/physiq_permute_e1.npy', {'inputs': i, 'labels': l, 'users': u})
    
    i, l, u, t = parser_folder_permuting('./datasets/physiq/segment_sessions_one_repetition_data_E2')
    np.save('./datasets/physiq/physiq_permute_e2.npy', {'inputs': i, 'labels': l, 'users': u, 'transition': t})
    
    # i, l, u = parser_folder_permuting('./datasets/physiq/segment_sessions_one_repetition_data_E3')
    # np.save('./datasets/physiq/physiq_permute_e3.npy', {'inputs': i, 'labels': l, 'users': u})
    
    i, l, u, t = parser_folder_permuting('./datasets/physiq/segment_sessions_one_repetition_data_E4')
    np.save('./datasets/physiq/physiq_permute_e4.npy', {'inputs': i, 'labels': l, 'users': u , 'transition': t})
    