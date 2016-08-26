import pandas as pd
import numpy as np

from manage_state import set_state, set_stand_state
from utilities import combine_csv, concat_data, blank_filter, resolve_acc_gyro
from rolltec_features import create_features

def combine_state_features(directory, state, window=40, stand=0):
    """
    convenience method to combine three steps in one function:
    (1) combine multiple csv files, (2) set their movement state for training,
    (3) detect any instances of standing up, (4) add features
    """
    
    combined_data = combine_csv(directory)
    combined_data_updated = set_state(combined_data, state)
    combined_data_updated2 = set_stand_state(combined_data_updated, stand)
    feature_training_data = create_features(combined_data_updated2, window)
    ready_training_data = set_state(feature_training_data, state)
    
    return ready_training_data
    

def prep(window=40):
    """prepare the raw sensor data
    the argument window determines the size of the sliding selection window
    for the time series. Given that data has been collected at a frequency of 
    25Hz, a sliding window of 40 will give you combined data windows 
    of 1.6 seconds.
    """

    #1 Your mount
    ymount_td = combine_state_features('your_mount_raw_data', 'your_mount', window, 0)
    #2 Your side control
    ysc_td = combine_state_features('your_side_control_raw_data', 'your_side_control', window, 0)
    #3 Your closed guard
    ycg_td = combine_state_features('your_closed_guard_raw_data', 'your_closed_guard', window, 0)
    #4 Your back control
    ybc_td = combine_state_features('your_back_control_raw_data', 'your_back_control', window, 0)
    #5 Opponent mount or opponent side control
    omountsc_td = combine_state_features('opponent_mount_and_opponent_side_control_raw_data', 'opponent_mount_or_sc', window, 0)
    #6 Opponent closed guard
    ocg_td = combine_state_features('opponent_closed_guard_raw_data', 'opponent_closed_guard', window, 0)
    #7 Opponent back control
    obc_td = combine_state_features('opponent_back_control_raw_data', 'opponent_back_control', window, 0)
    #8 "Non jiu-jitsu" motion
    nonjj_td = combine_state_features('non_jj_raw_data', 'non_jj', window, 0)
    #9 "stand up" motion
    stand_up_td = combine_state_features('standing_up_raw_data', 'opponent_closed_guard', window, 1)

    training_data = concat_data([ymount_td, ysc_td, ycg_td, ybc_td, omountsc_td, ocg_td, obc_td, nonjj_td, stand_up_td])
    # remove NaN
    training_data = blank_filter(training_data)
    return training_data
    
def prep_test(test_file):
    """ prepares test data to check for algorithm accuracy
    so does not set the state
    """
    
    el_file = 'data/test_cases/' + test_file
    df = pd.DataFrame()
    df = pd.read_csv(el_file, index_col=None, header=0)
    df = resolve_acc_gyro(df)
    df = create_features(df, _window=40, test=True)
    test_data = blank_filter(df)

    return test_data