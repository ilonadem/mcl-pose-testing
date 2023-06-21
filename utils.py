import os
import numpy as np
import csv
import glob

import pandas as pd
from ast import literal_eval

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# from sklearn.metrics import mean_squared_error

keypoints = ['NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EAR', 'RIGHT_EAR', 
             'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 
             'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 
             'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 
             'RIGHT_ANKLE']
# MCL collected less keypoints, so plotting is slightly modified
mcl_keypoints = ['LEFT_ANKLE', 'LEFT_EAR', 'LEFT_SHOULDER', 'LEFT_HIP',
                'LEFT_KNEE', 'RIGHT_ANKLE', 'RIGHT_EAR', 'RIGHT_SHOULDER', 
                 'RIGHT_HIP', 'RIGHT_KNEE', 'COM']

# indeces of keypoints between which we want to draw a line
num_edges = ((0, 1),(0, 2),(0, 3),(0, 4),(3, 1),(4, 2),
             (1, 2),(5, 6),(5, 7),(5, 11),(6, 8),(6, 12),
             (7, 9),(8, 10),(11, 12),(11, 13),(12, 14),
             (13, 15),(14, 16),)
kp_edges = ((1,6),(2,7),(2,3),(7,8),(3,8),(3,4),(8,9),(4,0),(9,5))


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
    f_df = pd.DataFrame(columns=keypoints, 
                        # dtype=np.int64
                        )
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

    df_full = create_df(data_files[0])
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
    # df['time'] = df.apply(lambda row: f"{((row.NOSE[-1][:2])-4):02}" + row.NOSE[-1][2:], axis=1)
    print()
    if type(df.iloc[0]['NOSE'][-1])==int:
        # df = add_times(df, [45 + (55.0/60), 46+(14.9/60)])
        df['time'] = df.apply(lambda row: row.NOSE[-2], axis=1)
    else:
        # if type(df.NOSE[0])
        df['time'] = df.apply(lambda row: float(str(row.NOSE[-2])[6:17]), axis=1)
    return df

def normalize_df(df, var1, var2):
    df[var1] = df[var1] - df[var1].mean()
    df[var1] = df[var1] / df[var1].abs().values.max()
    df[var2] = df[var2] - df[var2].mean()
    df[var2] = df[var2] / df[var2].abs().values.max() 
#     print(type(df[var1].max()))
    
    return df

def clean_keypoints(coral_df):
    coral_df['time_int'] = coral_df['time']
    return coral_df

def time_int_to_mins(df):
    df['time_int'] = df.apply(lambda row: int(row.time[3:5]) + float(row.time[6:])/(60), axis=1)
    
    return df

def time_correct(df):
    df['time_int'] = df.apply(lambda row: float(100*(row.time_int - int(row.time_int))), axis=1)
    return df

def read_movenet_kps(file):
    raw_df = pd.read_csv(file)

    for kp in keypoints:
        raw_df[kp] = raw_df[kp].apply(lambda x: literal_eval(x))
    
    raw_df['time'] = raw_df.apply(lambda row: row['NOSE'][-1], axis=1)

    return raw_df

def create_pose_df_from_file_folder(kp_dir):
    
    print("KP DIR: ", kp_dir)

    if 'movenet' in kp_dir:
        # movenet formatting is different from the others
        num_files = 0
        
        for file in os.listdir(kp_dir):
            if '.csv' in file: # assumes only one csv file in the folder
                if num_files == 0:
                    kp_df = read_movenet_kps(kp_dir + '/' + file)
                else:
                    df_new = read_movenet_kps(kp_dir + '/' + file)
                    kp_df = pd.concat([kp_df, df_new])
                num_files += 1
                
    else:
        kp_df = create_df_from_hour(kp_dir)
        kp_df = add_custom_time(kp_df)
    
    return kp_df

def scale_movenet_keypoints(kp_df):
    for kp in keypoints:
        # kp_df[f'{kp}_X'] = kp_df[f'{kp}_X'].apply(lambda x: x*640)
        # kp_df[f'{kp}_Y'] = kp_df[f'{kp}_Y'].apply(lambda x: x*480)
        kp_df[kp] = kp_df[kp].apply(lambda x: [x[1]*640, x[0]*480, x[2]])
    return kp_df

### MCL THINGS ###

def normalize_df(df, var1, var2):
    df[var1] = df[var1] - df[var1].mean()
    df[var1] = df[var1] / df[var1].abs().values.max()
    # df[var1] = df[var1] / -df[var1].values.min()
    df[var2] = df[var2] - df[var2].mean()
    df[var2] = df[var2] / df[var2].abs().values.max()
    # df[var2] = df[var2] / -df[var2].values.min() 
#     print(type(df[var1].max()))
    
    return df

def cleanup_mcl(df, mcl_keypoints):
    for var in mcl_keypoints:
    #     var = 'LEFT_ANKLE'
        df[var] = df.apply(lambda row: [row[f'{var}_X'], row[f'{var}_Y'], row[f'{var}_Z']], axis=1)
        del df[f'{var}_X']
        del df[f'{var}_Y']
        del df[f'{var}_Z']
    del df['ITEM']
    return df
    
def prep_plot_dfs(df1, df2, df3, plot_vars):
    # time correction
    time_int = [45 + (55.0/60), 46+(14.9/60)]
    df1 = add_times(df1, time_int)

    # # manually remove the outliers
    # df2 = df2[df2['RIGHT_HIP_Y']>100]
    # df3 = df3[df3['RIGHT_HIP_Y']>100]

    for plot_var in plot_vars:
        df1 = normalize_df(df1, f'{plot_var}_X', f'{plot_var}_Y')
        df2 = normalize_df(df2, f'{plot_var}_X', f'{plot_var}_Y')
        df3 = normalize_df(df3, f'{plot_var}_X', f'{plot_var}_Y')

    return df1, df2, df3

def models_compare(df1, df2, df3, plot_vars, plot_title):
    fig, ax = plt.subplots(nrows=len(plot_vars), ncols=2, figsize=(20,15))
    fig.suptitle(plot_title, fontsize=20)

    for i in range(len(plot_vars)):
        ax[i,0].plot(df1['time'], df1[plot_vars[i]+'_X'], label='mcl', color='black')
        ax[i,0].plot(df2['time'], df2[plot_vars[i]+'_X'], label='thunder', color='tab:blue')
        ax[i,0].plot(df3['time'], df3[plot_vars[i]+'_X'], label='lightning', color='green')
        ax[i,0].set_title(f'{plot_vars[i]} X')
        ax[i,0].set_xlabel('time (min)')
        ax[i,0].legend()

        ax[i,1].plot(df1['time'], df1[plot_vars[i]+'_Y'], label='mcl', color='black')
        ax[i,1].plot(df2['time'], df2[plot_vars[i]+'_Y'], label='thunder', color='tab:blue')
        ax[i,1].plot(df3['time'], df3[plot_vars[i]+'_Y'], label='lightning', color='green')
        ax[i,1].set_title(f'{plot_vars[i]} Y')
        ax[i,1].set_xlabel('time (min)')
        ax[i,1].legend()

    plt.show()

def scatter_compare(df1, df2, plot_vars, plot_title, df3=None, ):
    fig, ax = plt.subplots(nrows=len(plot_vars), ncols=2, figsize=(20,15))
    fig.suptitle(plot_title, fontsize=20)

    for i in range(len(plot_vars)):
        ax[i,0].scatter(df1['time'], df1[plot_vars[i]+'_X'], label='mcl', color='black')
        ax[i,0].scatter(df2['time'], df2[plot_vars[i]+'_X'], label='movenet thunder', color='tab:blue')
        if df3 is not None:
            ax[i,0].scatter(df3['time'], df3[plot_vars[i]+'_X'], label='lightning', color='green')
        ax[i,0].set_title(f'{plot_vars[i]} X')
        ax[i,0].set_xlabel('time (min)')
        ax[i,0].legend()

        ax[i,1].scatter(df1['time'], df1[plot_vars[i]+'_Y'], label='mcl', color='black')
        ax[i,1].scatter(df2['time'], df2[plot_vars[i]+'_Y'], label='movenet thunder', color='tab:blue')
        if df3 is not None:
            ax[i,1].scatter(df3['time'], df3[plot_vars[i]+'_Y'], label='lightning', color='green')
        ax[i,1].set_title(f'{plot_vars[i]} Y')
        ax[i,1].set_xlabel('time (min)')
        ax[i,1].legend()

    plt.show()

def make_3d_kps(df):
    for var in mcl_keypoints:
        df[var] = df.apply(lambda row: [row[f'{var}_X'], row[f'{var}_Y'], row[f'{var}_Z']], axis=1)
        del df[f'{var}_X']
        del df[f'{var}_Y']
        del df[f'{var}_Z'] 
    del df['ITEM']

def add_xyz(df):
    for var in mcl_keypoints:
        df[f'{var}_X'] = df.apply(lambda row: row[var][0], axis=1)
        df[f'{var}_Y'] = df.apply(lambda row: row[var][1], axis=1)
        df[f'{var}_Z'] = df.apply(lambda row: row[var][2], axis=1)
    return df

def rotate_mcl_kps(df, ang, t):
    z_rot = np.array([[np.cos(ang), -np.sin(ang), 0],
                      [np.sin(ang), np.cos(ang), 0],
                      [0, 0, 1.0]])

    for var in mcl_keypoints:
        df[var] = df.apply(lambda row: np.matmul(z_rot,row[var])+t, axis=1)
    
    return add_xyz(df)


def add_times(df, t):
    df['time'] = np.linspace(t[0], t[1], len(df))
    df['time_int'] = np.linspace(t[0], t[1], len(df))
    return df

def homogenize_kps(df):
    for var in mcl_keypoints:
        df[var] = df.apply(lambda row: np.append(row[f'{var}'], 1), axis=1)
    return df

def unhomogenize_kps(df):
    for var in mcl_keypoints:
        df[var] = df.apply(lambda row: row[f'{var}'][:-1], axis=1)
    return df

def rotate_kps(df, ang):
    rotation_matrix = np.array([[np.cos(ang), -np.sin(ang), 0],
                      [np.sin(ang), np.cos(ang), 0],
                      [0, 0, 1.0]])
    for var in mcl_keypoints:
        df[var] = df.apply(lambda row: np.matmul(rotation_matrix, row[f'{var}']), axis=1)
    return df

def transform_world2cam(df, extrinsic):
    for var in mcl_keypoints:
        df[var] = df.apply(lambda row: np.matmul(extrinsic, row[f'{var}']), axis=1)
    return df

def project(df, intrinsic):
    for var in mcl_keypoints:
        # df[var] = df.apply(lambda row: np.matmul(intrinsic, row[f'{var}']), axis=1)
        df[var] = df[var].apply( lambda x: np.matmul(intrinsic, x))
    return df

def transform_df(df, C_W, ang):
    ang = np.radians(ang) # assuming degrees were put in
    cam_2_world = np.array([[np.cos(ang), -np.sin(ang), 0],
                      [np.sin(ang), np.cos(ang), 0],
                      [0, 0, 1.0]])                    
    RC = np.matmul(cam_2_world, C_W).reshape(-1,1)
    extrinsic = np.concatenate((cam_2_world, RC), axis=1)
    df = homogenize_kps(df)
    df = transform_world2cam(df, extrinsic)
    # width, height = 640, 480
    width, height = 0, 0
    intrinsic = np.array([[-1.0, 0, width/2],
                            [0, -1.0, height/2],
                            [0, 0, -1.0]])
    # print(df.iloc[0]['LEFT_KNEE'])
    df = project(df, intrinsic)
    # print(df.iloc[0]['LEFT_KNEE'])
    df = add_xyz(df)
    return df