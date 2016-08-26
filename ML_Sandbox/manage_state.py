import pandas as pd

def set_state(df, state):
    """set the classification state for training"""

    if state == 'your_mount':
        df['state'] = 0
    elif state == 'your_side_control':
        df['state'] = 1
    elif state =='your_closed_guard':
        df['state'] = 2
    elif state =='your_back_control':
        df['state'] = 3
    elif state =='opponent_mount_or_sc':
        df['state'] = 4
    elif state =='opponent_closed_guard':
        df['state'] = 5
    elif state == 'opponent_back_control':
        df['state'] = 6
    elif state =='non_jj':
        df['state'] = 7
        
    return df


def set_stand_state(df, stand_state):
    """sets the particular state of the motion of standing up"""
    
    if (stand_state == 1):
        df['stand'] = 1
    else:
        df['stand'] = 0
        
    return df

def state_reconciler(df):
    """Find any instance of a standup
    return a list of all the states around that standup
    check if any of those states are YMOUNT
    if they are, change them to OCG
    """

    relevant_df = df[['avg_stand', 'state']]

    for i in range(-8, 9):
        title = 'shift{}'.format(i)
        relevant_df[title] = relevant_df.state.shift(i).fillna(False)
        
    detected_standup_df = relevant_df.loc[relevant_df['avg_stand'] == 1].apply(lambda x: x.tolist(), axis=1)
    surrounding_states = []
    surrounding_indexes = []

    for row in detected_standup_df.iterrows():
        index, data = row
        data_list = data.astype(int).tolist()
        data_list.pop(0) # first value is the 'avg_stand' feature which is not what we want
        data_list.pop(1) # 2nd value is the state which is not what we want
        surrounding_states.append(data_list)
        surrounding_indexes.append(index.tolist())

    
    # Now we check the surrounding states to see if there are any "ymount" values we
    # should convert to "ocg"
    new_values = []
    for index, sequence in enumerate(surrounding_states):
        for values in sequence:
            if (values == 0):
                new_values = [5 if x == 0 else x for x in sequence]
                actual_index = surrounding_indexes[index]
                relevant_df = update_df(df, actual_index, new_values)
                
    return relevant_df