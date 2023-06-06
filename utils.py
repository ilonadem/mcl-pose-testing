import os
import numpy as np
import csv
import glob

import pandas as pd
from ast import literal_eval

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sklearn.metrics import mean_squared_error

keypoints = ['NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EAR', 'RIGHT_EAR', 
             'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 
             'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 
             'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 
             'RIGHT_ANKLE']

# indeces of keypoints between which we want to draw a line
num_edges = ((0, 1),(0, 2),(0, 3),(0, 4),(3, 1),(4, 2),
             (1, 2),(5, 6),(5, 7),(5, 11),(6, 8),(6, 12),
             (7, 9),(8, 10),(11, 12),(11, 13),(12, 14),
             (13, 15),(14, 16),)

# same list but as keypoint names 
EDGES = (
    ('NOSE', 'LEFT_EYE'),
    ('NOSE', 'RIGHT_EYE'),
    ('NOSE', 'LEFT_EAR'),
    ('NOSE', 'RIGHT_EAR'),
    ('LEFT_EAR', 'LEFT_EYE'),
    ('RIGHT_EAR', 'RIGHT_EYE'),
    ('LEFT_EYE', 'RIGHT_EYE'),
    ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
    ('LEFT_SHOULDER', 'LEFT_ELBOW'),
    ('LEFT_SHOULDER', 'LEFT_HIP'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
    ('RIGHT_SHOULDER', 'RIGHT_HIP'),
    ('LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('LEFT_HIP', 'RIGHT_HIP'),
    ('LEFT_HIP', 'LEFT_KNEE'),
    ('RIGHT_HIP', 'RIGHT_KNEE'),
    ('LEFT_KNEE', 'LEFT_ANKLE'),
    ('RIGHT_KNEE', 'RIGHT_ANKLE'),
)

def create_df_from_file(file):
    print(file)
    raw_df = pd.read_csv(file)
#         print(raw_df)
    f_df = pd.DataFrame(columns=keypoints, dtype=np.int64)
    for kp in keypoints:
        f_df[kp] = literal_eval(raw_df[kp][0])

    return f_df

# todo: pass on empty files
def listdir(path):
    return glob.glob(os.path.join(path, '*'))

def create_df_from_hour(hourdir):

    def create_df(file):
        print(file)
        raw_df = pd.read_csv(file)
#         print(raw_df)
        f_df = pd.DataFrame(columns=keypoints, dtype=np.int64)
        for kp in keypoints:
            f_df[kp] = literal_eval(raw_df[kp][0])
        
        return f_df
            
    data_files = [file for file in listdir(hourdir) if '.csv' in file] 
    print("DATA FILES", data_files)
    data_files.sort()
    
    print(data_files)
    df_full = create_df(data_files[0])

    # if len(data_files) > 1:
    for f in data_files[1:]:
        df_new = create_df(f)
        df_full = pd.concat([df_full, df_new])
            
    return df_full

def crop_df_by_time(t_min, t_max, df):
    if t_min > 0:
        df = df[df['time_int'].values>=t_min]
    if t_max > 0:
        df = df[df['time_int'].values<=t_max]
    return df

def split_x_y(coral_df):
    # x and y vals
    for kp in keypoints:
        coral_df[f'{kp}_X'] = coral_df.apply(lambda row: np.float(row[kp][0]), axis=1)
        coral_df[f'{kp}_Y'] = coral_df.apply(lambda row: np.float(row[kp][1]), axis=1)
    #  delete clunk   
    for kp in keypoints:
        del coral_df[kp]
    return coral_df

def add_com(clean_coral_df):
    
    for kp in keypoints:
        clean_coral_df[f'{kp}_X'] = clean_coral_df.apply(lambda row: np.float(row[kp][0]), axis=1)
        clean_coral_df[f'{kp}_Y'] = clean_coral_df.apply(lambda row: np.float(row[kp][1]), axis=1)
        clean_coral_df[f'{kp}_PROB'] = clean_coral_df.apply(lambda row: np.float(row[kp][2]), axis=1)
    
    clean_coral_df['FOOT_X'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_ANKLE_X + 0.5*row.RIGHT_ANKLE_X, axis=1)
    clean_coral_df['FOOT_Y'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_ANKLE_X + 0.5*row.RIGHT_ANKLE_X, axis=1)

    clean_coral_df['LEFT_SHANK_X'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_KNEE_X + 0.5*row.LEFT_ANKLE_X, axis=1)
    clean_coral_df['LEFT_SHANK_Y'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_KNEE_Y + 0.5*row.LEFT_ANKLE_Y, axis=1)

    clean_coral_df['RIGHT_SHANK_X'] = clean_coral_df.apply(lambda row: 0.5*row.RIGHT_KNEE_X + 0.5*row.RIGHT_ANKLE_X, axis=1)
    clean_coral_df['RIGHT_SHANK_Y'] = clean_coral_df.apply(lambda row: 0.5*row.RIGHT_KNEE_Y + 0.5*row.RIGHT_ANKLE_Y, axis=1)

    clean_coral_df['SHANK_X'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_SHANK_X + 0.5*row.RIGHT_SHANK_X, axis=1)
    clean_coral_df['SHANK_Y'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_SHANK_Y + 0.5*row.RIGHT_SHANK_Y, axis=1)

    clean_coral_df['LEFT_THIGH_X'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_KNEE_X + 0.5*row.LEFT_HIP_X, axis=1)
    clean_coral_df['LEFT_THIGH_Y'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_KNEE_Y + 0.5*row.LEFT_HIP_Y, axis=1)
    clean_coral_df['RIGHT_THIGH_X'] = clean_coral_df.apply(lambda row: 0.5*row.RIGHT_KNEE_X + 0.5*row.RIGHT_HIP_X, axis=1)
    clean_coral_df['RIGHT_THIGH_Y'] = clean_coral_df.apply(lambda row: 0.5*row.RIGHT_KNEE_Y + 0.5*row.RIGHT_HIP_Y, axis=1)

    clean_coral_df['THIGH_X'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_THIGH_X + 0.5*row.RIGHT_THIGH_X, axis=1)
    clean_coral_df['THIGH_Y'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_THIGH_Y + 0.5*row.RIGHT_THIGH_Y, axis=1)

    clean_coral_df['HAND_X'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_WRIST_X + 0.5*row.RIGHT_WRIST_X, axis=1)
    clean_coral_df['HAND_Y'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_WRIST_Y + 0.5*row.RIGHT_WRIST_Y, axis=1)

    clean_coral_df['LEFT_FOREARM_X'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_WRIST_X + 0.5*row.LEFT_ELBOW_X, axis=1)
    clean_coral_df['LEFT_FOREARM_Y'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_WRIST_Y + 0.5*row.LEFT_ELBOW_Y, axis=1)
    clean_coral_df['RIGHT_FOREARM_X'] = clean_coral_df.apply(lambda row: 0.5*row.RIGHT_WRIST_X + 0.5*row.RIGHT_ELBOW_X, axis=1)
    clean_coral_df['RIGHT_FOREARM_Y'] = clean_coral_df.apply(lambda row: 0.5*row.RIGHT_WRIST_Y + 0.5*row.RIGHT_ELBOW_Y, axis=1)

    clean_coral_df['FOREARM_X'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_WRIST_X + 0.5*row.RIGHT_WRIST_X, axis=1)
    clean_coral_df['FOREARM_Y'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_WRIST_Y + 0.5*row.RIGHT_WRIST_Y, axis=1)

    clean_coral_df['LEFT_UPARM_X'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_SHOULDER_X + 0.5*row.LEFT_ELBOW_X, axis=1)
    clean_coral_df['LEFT_UPARM_Y'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_SHOULDER_Y + 0.5*row.LEFT_ELBOW_Y, axis=1)
    clean_coral_df['RIGHT_UPARM_X'] = clean_coral_df.apply(lambda row: 0.5*row.RIGHT_SHOULDER_X + 0.5*row.RIGHT_ELBOW_X, axis=1)
    clean_coral_df['RIGHT_UPARM_Y'] = clean_coral_df.apply(lambda row: 0.5*row.RIGHT_SHOULDER_Y + 0.5*row.RIGHT_ELBOW_Y, axis=1)

    clean_coral_df['UPARM_X'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_UPARM_X + 0.5*row.RIGHT_UPARM_X, axis=1)
    clean_coral_df['UPARM_Y'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_UPARM_Y + 0.5*row.RIGHT_UPARM_Y, axis=1)

    clean_coral_df['PELVIS_X'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_HIP_X + 0.5*row.RIGHT_HIP_X, axis=1)
    clean_coral_df['PELVIS_Y'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_HIP_Y + 0.5*row.RIGHT_HIP_Y, axis=1)

    clean_coral_df['LEFT_THORAX_X'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_SHOULDER_X + 0.5*row.LEFT_HIP_X, axis=1)
    clean_coral_df['LEFT_THORAX_Y'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_SHOULDER_Y + 0.5*row.LEFT_HIP_Y, axis=1)
    clean_coral_df['RIGHT_THORAX_X'] = clean_coral_df.apply(lambda row: 0.5*row.RIGHT_SHOULDER_X + 0.5*row.RIGHT_HIP_X, axis=1)
    clean_coral_df['RIGHT_THORAX_Y'] = clean_coral_df.apply(lambda row: 0.5*row.RIGHT_SHOULDER_Y + 0.5*row.RIGHT_HIP_Y, axis=1)

    clean_coral_df['THORAX_X'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_THORAX_X + 0.5*row.RIGHT_THORAX_X, axis=1)
    clean_coral_df['THORAX_Y'] = clean_coral_df.apply(lambda row: 0.5*row.LEFT_THORAX_Y + 0.5*row.RIGHT_THORAX_Y, axis=1)

    clean_coral_df['HEAD_X'] = clean_coral_df['NOSE_X']
    clean_coral_df['HEAD_Y'] = clean_coral_df['NOSE_Y']

    clean_coral_df['TRUNK_X'] = clean_coral_df['THORAX_X']
    clean_coral_df['TRUNK_Y'] = clean_coral_df['THORAX_Y']

    clean_coral_df['COM_X'] = clean_coral_df.apply(lambda row: (1/1.286)*(0.0145*row.FOOT_X + 
                                                                          0.0465*row.SHANK_X + 
                                                                          0.10*row.THIGH_X + 
                                                                          0.006*row.HAND_X + 
                                                                          0.016*row.FOREARM_X + 
                                                                          0.028*row.UPARM_X + 
                                                                          0.142*row.PELVIS_X + 
                                                                          0.355*row.THORAX_X + 
                                                                          0.081*row.HEAD_X + 
                                                                          0.497*row.TRUNK_X) , axis=1)

    clean_coral_df['COM_Y'] = clean_coral_df.apply(lambda row: (1/1.286)*(0.0145*row.FOOT_Y + 
                                                                          0.0465*row.SHANK_Y + 
                                                                          0.10*row.THIGH_Y + 
                                                                          0.006*row.HAND_Y + 
                                                                          0.016*row.FOREARM_Y + 
                                                                          0.028*row.UPARM_Y + 
                                                                          0.142*row.PELVIS_Y + 
                                                                          0.355*row.THORAX_Y + 
                                                                          0.081*row.HEAD_Y + 
                                                                          0.497*row.TRUNK_Y) , axis=1)
    
    return clean_coral_df

def add_custom_time(df):
    print(df['NOSE'][1][-1])
    print(df.keys())
    # df['time'] = df.apply(lambda row: f"{((row.NOSE[-1][:2])-4):02}" + row.NOSE[-1][2:], axis=1)
    df['time'] = df.apply(lambda row: float(str(row.NOSE[-1])[6:17]), axis=1)
    print(df['time'])
    return df

def normalize_df(df, var1, var2):
    df[var1] = df[var1] - df[var1].mean()
    df[var1] = df[var1] / df[var1].abs().values.max()
    df[var2] = df[var2] - df[var2].mean()
    df[var2] = df[var2] / df[var2].abs().values.max() 
#     print(type(df[var1].max()))
    
    return df

def clean_keypoints(coral_df):
    # x and y vals
#     for kp in keypoints:
#         coral_df[f'{kp}_X'] = coral_df.apply(lambda row: np.float(row[kp][0]), axis=1)
#         coral_df[f'{kp}_Y'] = coral_df.apply(lambda row: np.float(row[kp][1]), axis=1)
#     #  delete clunk   
#     for kp in keypoints:
#         del coral_df[kp]
        
    # coral_df['time_int'] = coral_df.apply(lambda row: int(str(row.time)[:2]) + int(str(row.time)[3:5])/10e1 + int(str(row.time)[6:8])/10e3 + int(str(row.time)[9:])/10e9, axis=1)
    # coral_df['time_int'] = coral_df['time_int'].round(5)
    coral_df['time_int'] = coral_df['time']
    return coral_df

def time_int_to_mins(df):
    df['time_int'] = df.apply(lambda row: int(row.time[3:5]) + float(row.time[6:])/(60), axis=1)
    
    return df

def time_correct(df):
    # df['time_int'] = df.apply(lambda row: float('0'+str(row.time_int)[2:]), axis=1)
    print(df['time_int'])
    df['time_int'] = df.apply(lambda row: float(100*(row.time_int - int(row.time_int))), axis=1)
    print(df['time_int'])
    return df