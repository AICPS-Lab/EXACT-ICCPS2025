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
        # in this case, label should a (N x window_size x 6), and after this func, the label would be aggregate (bincount) to be classification problem
        transitions = []
        for each_window in label:
            if len(np.unique(each_window)) > 1:
                transitions.append(1)
            else:
                transitions.append(0)
        return transitions
def is_overlap_within_threshold(total_elements, begin_index, end_index, target_third, threshold=.75):
    # Define the boundaries of each third
    one_third = total_elements / 3
    two_thirds = 2 * one_third
    
    # Calculate the start and end of the target third
    if target_third == "begin":
        target_start = 0
        target_end = one_third
    elif target_third == "mid":
        target_start = one_third
        target_end = two_thirds
    elif target_third == "end":
        target_start = two_thirds
        target_end = total_elements
    else:
        raise ValueError("Invalid target_third. Choose 'begin', 'mid', or 'end'.")

    # Calculate the overlap with the target third
    overlap_start = max(target_start, begin_index)
    overlap_end = min(target_end, end_index)
    overlap = max(0, overlap_end - overlap_start)
    
    # Calculate the total range covered by begin_index to end_index
    total_range = end_index - begin_index
    
    # Calculate the overlap percentage
    overlap_percentage = (overlap / total_range)  if total_range > 0 else 0
    
    # Check if the overlap percentage meets or exceeds the threshold
    return overlap_percentage >= threshold

def find_mid(label, window_size, step_size, dense_label=False):
    if dense_label:
        # get the length of exercises (based on the change of label):
        length_exercises = [0]
        prev_label = label[0]
        # in case an example of this happened: 0, 1, 0 ... 0, but should also work for 0 0 0 1 1 1
        for i in range(1, len(label)-1):
            
            cur_label = label[i]
            next_label = label[i+1]
            if cur_label != prev_label:
                if cur_label != next_label:
                    prev_label = cur_label
                    continue
                else:
                    length_exercises.append(i)
                    prev_label = cur_label
        def index_at(i):
            prev_each = length_exercises[0]
            for each in length_exercises[1::]:
                if prev_each <= i < each:
                    return prev_each, each, each - prev_each
                prev_each = each

            return -1, -1, -1
        
        # find all the middle within the exercise:
        transitions = []
        cur_count = 0
        index = 0
        for i in range(0, len(label), step_size):
            window = label[i:i+window_size]
            if len(window) == window_size:
                total_before, is_at, cur_exercise_length = index_at(i)
                cur_exercise_at = i - total_before
                assert cur_exercise_at >=0, "The index is not within the exercise, cur_exercise_at :{}, current i {}".format(cur_exercise_at, i)
                if is_overlap_within_threshold(cur_exercise_length, cur_exercise_at, cur_exercise_at + window_size, "mid"):
                    transitions.append(1)
                else:
                    transitions.append(0)
        return transitions
    else:
        raise NotImplementedError("Would not work for classification problem, only for dense label problem.")
    

                    

        
    
    
