## Testing Pose Models on MCL Data

Based on PoseNet infrastructure of this repository is based on this repo: (https://github.com/rwightman/posenet-python)

### Install

To install the required packages in a conda environment:

```
conda env create -f env.yml
conda activate tf2
```

A suitable Python 3.x environment with a recent version of Tensorflow is required.

This relies on an older version of tensorflow, so it requires disabling tensorflow v2 behavior in any scripts:
```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
``` 

<!-- Development and testing was done with Conda Python 3.6.8 and Tensorflow 1.12.0 on Linux.

Windows 10 with the latest (as of 2019-01-19) 64-bit Python 3.7 Anaconda installer was also tested. -->

If you want to use the webcam demo, a pip version of opencv (`pip install opencv-python`) is required instead of the conda version. Anaconda's default opencv does not include ffpmeg/VideoCapture support. Also, you may have to force install version 3.4.x as 4.x has a broken drawKeypoints binding.

A conda environment setup as below should suffice: 
```
conda install tensorflow-gpu scipy pyyaml python=3.6
pip install opencv-python==3.4.5.20

```

### Usage

There are three demo apps in the root that utilize the PoseNet model. They are very basic and could definitely be improved.

The first time these apps are run (or the library is used) model weights will be downloaded from the TensorFlow.js version and converted on the fly.

For all demos, the model can be specified with the '--model` argument by using its ordinal id (0-3) or integer depth multiplier (50, 75, 100, 101). The default is the 101 model.

#### image_demo.py 

Image demo runs inference on an input folder of images and outputs those images with the keypoints and skeleton overlayed.

`python image_demo.py --model 101 --image_dir ./images --output_dir ./output`

A folder of suitable test images can be downloaded by first running the `get_test_images.py` script.

#### benchmark.py

A minimal performance benchmark based on image_demo. Images in `--image_dir` are pre-loaded and inference is run `--num_images` times with no drawing and no text output.

#### webcam_demo.py

The webcam demo uses OpenCV to capture images from a connected webcam. The result is overlayed with the keypoints and skeletons and rendered to the screen. The default args for the webcam_demo assume device_id=0 for the camera and that 1280x720 resolution is possible.

### Credits

The PoseNet and model infra structure of this repository is based on this repo: (https://github.com/rwightman/posenet-python)

The original model, weights, code, etc. was created by Google and can be found at https://github.com/tensorflow/tfjs-models/tree/master/posenet





