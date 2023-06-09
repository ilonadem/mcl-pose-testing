from movenet_utils import *

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2
import os
import csv

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import pandas as pd

# Some modules to display an animation using imageio.
import imageio
from IPython.display import HTML, display

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, default='../keypoint_files/movenet', help='specify directory to save keypoint csvs')
parser.add_argument('--title', type=str, default='movenet', help='specify title of video')
args = parser.parse_args()

def movenet(input_image):
    """Runs detection on an input image.

    Args:
        input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
    """
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

model_name = [
    'movenet_thunder',
    # 'movenet_lightning',
]

# import the model
if "movenet_lightning" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    input_size = 192
elif "movenet_thunder" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    input_size = 256
else:
    raise ValueError("Unsupported model name: %s" % model_name)

# Initialize the video capture object
mcl_video_file = '/Users/ilonademler/Documents/Harvard/seniorfall/coral/mcl_experiments/mcl_data/MCL_4_14/videos/test.mp4'

cap = cv2.VideoCapture(mcl_video_file)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

frame_count = 0
while(cap.isOpened()):
  ret, frame = cap.read()
  frame = np.expand_dims(frame, axis=0)
  frame_count += 1
  if ret == True:
    if frame_count == 1:
      frames = frame
    elif frame_count < 10:
      frames = np.concatenate((frames, frame), axis=0)
  else:
    break

image = tf.convert_to_tensor(frames)

# Load the input image.
num_frames, image_height, image_width, _ = image.shape
crop_region = init_crop_region(image_height, image_width)

# output_images = []
kp_dict = create_kp_dict(keypoints)

bar = display(progress(0, num_frames-1), display_id=True)
for frame_idx in range(num_frames):
  keypoints_with_scores = run_inference(
      movenet, image[frame_idx, :, :, :], crop_region,
      crop_size=[input_size, input_size])
  kp_dict = update_kp_dict(kp_dict, pose_dict_from_kps(keypoints_with_scores))

interval = [45 + 55/60, 46 + 16/60]
kp_df = pd.DataFrame(kp_dict, columns=keypoints)
add_time_to_mcl_data(interval, kp_df)
for kp in keypoints:
  kp_df[kp] = kp_df.apply(lambda row: np.concatenate((row[kp],[row['time']]),axis=0).astype(float), axis=1)
  kp_df[kp] = kp_df.apply(lambda row: row[kp].tolist(), axis=1)
del kp_df['time']

# save as .csv
print("saving in directory ", args.savedir)
os.makedirs(args.savedir, exist_ok=True)
csv_file = args.savedir + f'/{args.title}_poses.csv'
kp_df.to_csv(csv_file, mode='w', index=False)

# for some reason this doesn't work with this version of python :-[
# kp_df.to_csv(csv_file)
# os.makedirs(csv_file)
# with open(csv_file, 'w') as f:
#     writer = csv.DictWriter(f, kp_dict.keys())
#     writer.writeheader()
#     writer.writerow(kp_dict)
