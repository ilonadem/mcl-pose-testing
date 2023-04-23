################################################

# import tensorflow as tf
import cv2
import time
import argparse
import numpy as np
import posenet
import csv

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

poses_list = ['NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EAR', 'RIGHT_EAR', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']
poses_dict = {key: [] for key in poses_list}

def _process_input(source_img, scale_factor=1.0, output_stride=16):
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.reshape(1, target_height, target_width, 3)
    return input_img, source_img, scale

def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height

def main():
    video_files = ['cam1_2', 'cam2_2', 'cam3_2', 'cam4_2']
    # video_files = ['recording1', 'recording2', 'recording3', 'recording4']
    
    for video_file in video_files:
        start_time, end_time = 10.51 + (40/6000), 10.51 + (48/6000)

        with tf.Session() as sess:
            model_cfg, model_outputs = posenet.load_model(args.model, sess)
            output_stride = model_cfg['output_stride']

            # if args.file is not None:
            #     cap = cv2.VideoCapture(args.file)
            # else:
            #     cap = cv2.VideoCapture(args.cam_id)
            

            # Create a VideoCapture object and read from input file
            # If the input is the camera, pass 0 instead of the video file name
            cap = cv2.VideoCapture(f'videos/{video_file}.mp4')

            # Check if camera opened successfully
            if (cap.isOpened()== False): 
                print("Error opening video stream or file")

            cap.set(3, args.cam_width)
            cap.set(4, args.cam_height)

            # start = time.time()
            frame_count = 0
            while(cap.isOpened()):

                ret, frame = cap.read()
                frame_count += 1
                # input_image, display_image, output_scale = posenet.read_cap(
                #     cap, scale_factor=args.scale_factor, output_stride=output_stride)

                if ret == True:
                    input_image, display_image, output_scale = _process_input(frame)
                    
                    
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                    
                    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                            model_outputs,
                            feed_dict={'image:0': input_image}
                        )
                    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                    heatmaps_result.squeeze(axis=0),
                    offsets_result.squeeze(axis=0),
                    displacement_fwd_result.squeeze(axis=0),
                    displacement_bwd_result.squeeze(axis=0),
                    output_stride=output_stride,
                    max_pose_detections=1,
                    min_pose_score=0.15)

                    keypoint_coords *= output_scale
                    pose = keypoint_coords[0,:]
                    scores = keypoint_scores[0,:]
                    # print(f"pose w shape {pose.shape}: \n", pose)
                    # print(f"scores w shape {scores.shape}: \n", scores)
                    for i in range(len(poses_list)):
                        kp = poses_list[i]
                        kp_coords = pose[i]
                        kp_score = scores[i]

                        # print(f'kp: {kp}, kp coords: {kp_coords}, kp score: {kp_score}')
                
                        poses_dict[kp].append([kp_coords[1], kp_coords[0], kp_score])
            
            
                # print("pose scores shape: ", pose_scores.shape)
                # print("keypoint scores shape: ", keypoint_scores.shape)
                # print(keypoint_scores.shape)
                # print(keypoint_coords[0,:].shape

                else:
                    break

                # input_image, display_image, output_scale = posenet.read_cap(
                #     cap, scale_factor=args.scale_factor, output_stride=output_stride)

                
                

                # # TODO this isn't particularly fast, use GL for drawing and display someday...
                # overlay_image = posenet.draw_skel_and_kp(
                #     display_image, pose_scores, keypoint_scores, keypoint_coords,
                #     min_pose_score=0.15, min_part_score=0.1)

                # cv2.imshow('posenet', overlay_image)
                # frame_count += 1
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                # else:
                #     break
            
            cap.release()
            cv2.destroyAllWindows()

            # print('Average FPS: ', frame_count / (time.time() - start))
            csv_file = f'videos/{video_file}_df.csv'
            print("saving csv file name: ", csv_file)

            times = np.linspace(start_time, end_time, frame_count+1)
            poses_dict['time_int'] = times

            with open(csv_file, 'w') as f:
                writer = csv.DictWriter(f, poses_dict.keys())
                writer.writeheader()
                writer.writerow(poses_dict)
                
                # break

        


if __name__ == "__main__":
    main()