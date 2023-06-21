# Script for generating a keypoint animation from a keypoint dataframe

import os
import numpy as np
import csv
import pandas as pd
import glob
import re

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

from utils import *
from functools import partial

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--keypoint_folder', default='keypoint_files/patient_kps', type=str, help='specify file containing keypoint csvs')
parser.add_argument('--savedir', default='keypoint_files/visualizations', type=str, help='specify directory to save video')
parser.add_argument('--show', type=bool, default=False)
parser.add_argument('--start', type=float, default=0)
parser.add_argument('--end', type=float, default=-1)
args = parser.parse_args()

def init():
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    return ax

def animate(frame, coral_df, fig, ax, detection_threshold=0.0):
    
    # animates a single frame
    timestamp = times[frame]
    
    poses = coral_df.loc[coral_df['time_int']==timestamp]
    image = poses['image']
    
    ax.clear()
    # ax.set_title(f"{month}/{day}/{year}")
    for pos in range(len(poses)):
        pose = poses.iloc[pos].to_dict()
    
        width, height = 640, 480
        all_x_coords, all_y_coords, probs = [], [], []
        x_coords, y_coords = [], []

        for kp in keypoints:
            all_x_coords.append(pose[kp][0])
            all_y_coords.append(pose[kp][1])
            probs.append(pose[kp][2])

            if pose[kp][2] > detection_threshold:
                x_coords.append(pose[kp][0])
                y_coords.append(pose[kp][1])
        ax.scatter(x=x_coords, y=y_coords)
        ax.text(430.0, 20.0, f"time: {np.round(timestamp,4)}")
        ax.text(430.0, 60.0, f"frame: {frame}")
        ax.text(430.0, 100.0, f"image: {image}")
        for i,j in num_edges:
            if probs[i] > detection_threshold and probs[j] > detection_threshold:
                xs, ys = [all_x_coords[i], all_x_coords[j]], [all_y_coords[i], all_y_coords[j]]
                ax.plot(xs, ys, color='black')
        ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    
    return ax

# create dataframe
# if 'movenet' in args.keypoint_folder:
#     # movenet formatting is different from the others
#     print("os.listdir(args.keypoint_folder)", os.listdir(args.keypoint_folder))
#     for file in os.listdir(args.keypoint_folder):
#         if '.csv' in file:
#             kp_df = read_movenet_kps(args.keypoint_folder + '/' + file)
# else:
#     kp_df = create_df_from_hour(args.keypoint_folder)
#     kp_df = add_custom_time(kp_df)
kp_df = create_pose_df_from_file_folder(args.keypoint_folder)
    
# clean dataframe
kp_df = clean_keypoints(kp_df)
kp_df = crop_df_by_time(args.start, args.end, kp_df)
filename = args.keypoint_folder.split('/')[-1]
if 'movenet' in args.keypoint_folder:
    filename = 'movenet' + args.keypoint_folder.split('/')[-2] + "_" + filename

# scale movenet keypoints:
if 'movenet' in args.keypoint_folder:
    print("rescaling!")
    kp_df = scale_movenet_keypoints(kp_df)

# generate animations
fig, ax = plt.subplots()
width, height = 640, 480

coral_dataframes = [kp_df.copy()]

print(kp_df.head())
print("coral dataframes")
print("length of kp df", len(kp_df))
print(kp_df['time'].to_numpy().max(), kp_df['time'].to_numpy().min())
print("length of coral dataframes: ", len(coral_dataframes))
print(coral_dataframes)

# for i in range(len(coral_dataframes)):
# for i in range(len(kp_df.tolist())):
#     coral_df = kp_df.iloc[i]
for i in range(1):
    coral_df = kp_df
    
    times = coral_df['time'].unique()
    ani = FuncAnimation(fig, partial(animate, coral_df=coral_df, fig=fig, ax=ax), frames=len(times)-1, init_func=init)
    if args.show:
        plt.show()
    
    # save animation:
    vid_title = f'{args.savedir}/{filename}/{filename}_anim.mp4'
    os.makedirs(f'{args.savedir}/{filename}', exist_ok=True)
    writervideo = FFMpegWriter(fps=20)
    print("saving to", vid_title)
    ani.save(vid_title, writer=writervideo)