import math
import pandas as pd
import numpy as np

from manage_state import set_state, set_stand_state
from utilities import combine_csv

def root_sum_square(x, y, z):
        sum = ((x**2)+(y**2)+(z**2))
        rss = math.sqrt(sum)
        return rss

def root_mean_square(x, y, z):
        mean = ((x**2)+(y**2)+(z**2))/3
        rss = math.sqrt(mean)
        return rss

def tiltx(x, y, z):
    try:
        prep = (x/(math.sqrt((y**2)+(z**2))))
        tilt = math.atan(prep)
    except ZeroDivisionError:
        tilt = 0
    return tilt

def tilty(x, y, z):
    try:
        prep = (y/(math.sqrt((x**2)+(z**2))))
        tilt = math.atan(prep)
    except ZeroDivisionError:
        tilt = 0
    return tilt
    
def max_min_diff(max, min):
    diff = max - min
    return diff

def magnitude(x, y, z):
    magnitude = x + y + z
    return magnitude

def create_features(df, _window=40, test=False, label_test=False):
    """builds the data features, then applies
    overlapping logic
    """
    
    TIME_SEQUENCE_LENGTH = _window
    
    accel_x = df['ACCEL_X'].astype(float)
    accel_y = df['ACCEL_Y'].astype(float)
    accel_z = df['ACCEL_Z'].astype(float)
    gyro_x = df['GYRO_X'].astype(float)
    gyro_y = df['GYRO_Y'].astype(float)
    gyro_z = df['GYRO_Z'].astype(float)
    
    df2 = pd.DataFrame()
    
    # capture tilt here, then average later  
    df2['tiltx'] = df.apply(lambda x: tiltx(x['ACCEL_X'], x['ACCEL_Y'], x['ACCEL_Z']), axis=1)
    df2['tilty'] = df.apply(lambda x: tilty(x['ACCEL_X'], x['ACCEL_Y'], x['ACCEL_Z']), axis=1)

    # If it is a labelled data set, we will keep the state info
    if label_test == True:
        df2['state'] = df['state'].astype(int)
    
    # Capture stand state here, then average later
    if (test==False):
        df2['stand'] = df['stand'].astype(float)
    
    # Set 3 axes values to rolling mean for accelerometer and gyroscope
    for col, element in zip([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z],  ["ACCEL_X","ACCEL_Y","ACCEL_Z","GYRO_X","GYRO_Y","GYRO_Z"]):
        df2[element] = pd.rolling_mean(col, TIME_SEQUENCE_LENGTH-2, center=True)
    
    # rolling median
    for col, element in zip([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z],  ["x","y","z","gx","gy","gz"]):
        df2['rolling_median_'+element] = pd.rolling_median(col, TIME_SEQUENCE_LENGTH-2, center=True)
    
    # rolling max
    for col, element in zip([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z],  ["x","y","z","gx","gy","gz"]):
        df2['rolling_max_'+element] = pd.rolling_max(col, TIME_SEQUENCE_LENGTH-2, center=True)
    
    # rolling min
    for col, element in zip([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z],  ["x","y","z","gx","gy","gz"]):
        df2['rolling_min_'+element] = pd.rolling_min(col, TIME_SEQUENCE_LENGTH-2, center=True)
    
    # rolling sum
    for col, element in zip([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z],  ["x","y","z","gx","gy","gz"]):
        df2['rolling_sum_'+element] = pd.rolling_sum(col, TIME_SEQUENCE_LENGTH-2, center=True)
    
    # standard deviation
    for col, element in zip([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z],  ["x","y","z","gx","gy","gz"]):
        df2['rolling_std_'+element] = pd.rolling_std(col, TIME_SEQUENCE_LENGTH-2, center=True)
    
    # Tilt
    df2['avg_tiltx'] = pd.rolling_mean(df2['tiltx'], TIME_SEQUENCE_LENGTH-2, center=True)
    df2['avg_tilty'] = pd.rolling_mean(df2['tilty'], TIME_SEQUENCE_LENGTH-2, center=True)
    
    if (test==False):
        # standing up detection
        df2['avg_stand'] = pd.rolling_mean(df2['stand'], TIME_SEQUENCE_LENGTH-2, center=True)

        # round standing up as we need it to be either '0' or '1' for training later
        df2['avg_stand'] = df2['avg_stand'].apply(lambda x: math.ceil(x))

    ol_upper = _window/2
    ol_lower = ol_upper-1
    
    print "df2 length: {}".format(len(df2))
    # 50% overlap of the windows - very important
    # sliding_df = df2[ol_lower::ol_upper].copy() 
    print "sliding df length: {}".format(len(df2))
    
    df2['max_min_x'] = df2.apply(lambda x: max_min_diff(x['rolling_max_x'], x['rolling_min_x']), axis=1)
    df2['max_min_y'] = df2.apply(lambda x: max_min_diff(x['rolling_max_y'], x['rolling_min_y']), axis=1)
    df2['max_min_z'] = df2.apply(lambda x: max_min_diff(x['rolling_max_z'], x['rolling_min_z']), axis=1)
    df2['max_min_gx'] = df2.apply(lambda x: max_min_diff(x['rolling_max_gx'], x['rolling_min_gx']), axis=1)
    df2['max_min_gy'] = df2.apply(lambda x: max_min_diff(x['rolling_max_gy'], x['rolling_min_gy']), axis=1)
    df2['max_min_gz'] = df2.apply(lambda x: max_min_diff(x['rolling_max_gz'], x['rolling_min_gz']), axis=1)
                                                                       
    df2['acc_rss'] = df2.apply(lambda x: root_sum_square(x['ACCEL_X'], x['ACCEL_Y'], x['ACCEL_Z']), axis=1)
    df2['gyro_rss'] = df2.apply(lambda x: root_sum_square(x['GYRO_X'], x['GYRO_Y'], x['GYRO_Z']), axis=1)
    
    df2['acc_rms'] = df2.apply(lambda x: root_mean_square(x['ACCEL_X'], x['ACCEL_Y'], x['ACCEL_Z']), axis=1)
    df2['gyro_rms'] = df2.apply(lambda x: root_mean_square(x['GYRO_X'], x['GYRO_Y'], x['GYRO_Z']), axis=1)
    
    df2['acc_magnitude'] = df2.apply(lambda x: magnitude(x['ACCEL_X'], x['ACCEL_Y'], x['ACCEL_Z']), axis=1)
    df2['gyro_magnitude'] = df2.apply(lambda x: magnitude(x['GYRO_X'], x['GYRO_Y'], x['GYRO_Z']), axis=1)
        
    return df2
    
