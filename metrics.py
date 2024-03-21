import numpy as np

def find_transitions(label, window_size, step_size, dense_label=False):
    if dense_label:
        # in the sliding windows, with step size, we need to find if theres a transition in the windows:
        transitions = []
        # do sliding window on label:
        for i in range(0, len(label), step_size):
            window = label[i:i+window_size]
            if len(window) == window_size:
                # check if there's a transition:
                if len(np.unique(window)) > 1:
                    transitions.append(1)
                else:
                    transitions.append(0)
        return transitions
    else:
        raise NotImplementedError

def find_middle(label, window_size, step_size, dense_label=False):
    if dense_label:
        # in the sliding windows, with step size, we need to find if theres a transition in the windows:
        middle = []
        # do sliding window on label:
        for i in range(0, len(label), step_size):
            window = label[i:i+window_size]
            if len(window) == window_size:
                # check if there's a transition:
                middle.append(window[window_size//2])
        return middle
    else:
        raise NotImplementedError
    

                    

        
    
    
