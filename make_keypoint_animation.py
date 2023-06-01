# Script for generating a keypoint animation from a keypoint dataframe

import os
import numpy as np
import csv
import pandas as pd
import glob

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter

from utils import *
from functools import partial

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--keypoint_folder', type=str, help='specify file containing keypoint csvs')
parser.add_argument('--title', type=str)
args = parser.parse_args()

def init():
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    # ax.set_title(f"{month}/{day}/{year}")
    return ax

def animate(frame, coral_df, fig, ax, detection_threshold=0.0):
    
    # animates a single frame
    timestamp = times[frame]
    
    poses = coral_df.loc[coral_df['time_int']==timestamp]
    
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
#         ax.scatter(x=x_coords, y=y_coords, color='blue')
        ax.text(430.0, 20.0, f"time: {np.round(timestamp,4)}")
        ax.text(430.0, 60.0, f"frame: {frame}")
        for i,j in num_edges:
            if probs[i] > detection_threshold and probs[j] > detection_threshold:
                xs, ys = [all_x_coords[i], all_x_coords[j]], [all_y_coords[i], all_y_coords[j]]
                ax.plot(xs, ys, color='black')
        ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    
    return ax

kp_df = create_df_from_hour(args.keypoint_folder)
kp_df = add_custom_time(kp_df)
kp_df = clean_keypoints(kp_df)

# GENERATE ANIMATIONS 
fig, ax = plt.subplots()
width, height = 640, 480

coral_dataframes = [kp_df.copy()]
file_titles = ['test_animation']

for i in range(len(coral_dataframes)):
    coral_df = coral_dataframes[i]
    vid_title = args.keypoint_folder + '/' + args.title + '.mp4'
    # coral_df = c1_df.copy()
    times = coral_df['time_int'].unique()
    # ani = FuncAnimation(fig, animate, interval=10)
    ani = FuncAnimation(fig, partial(animate, coral_df=coral_df, fig=fig, ax=ax), frames=len(times)-1, init_func=init)
    # ani = FuncAnimation(fig, partial(animate2, coral_df=coral_df, fig=fig, ax=ax), interval=len(coral_df)-3, save_count=len(coral_df)-3, repeat=False)
    # plt.show()
    
    # save animation:
    writervideo = FFMpegWriter(fps=20)
    ani.save(vid_title, writer=writervideo)
