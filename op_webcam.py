import sys
import cv2
import argparse
import numpy as np
import time

sys.path.append('/usr/local/python')

from openpose import pyopenpose as op

# Flags
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='./openpose/models/')
parser.add_argument('--target_image', type=str, default='./data/warrior1/warrior1-0.png')
parser.add_argument('--net_resolution', type=str, default='176x192')
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--number_people_max', type=int, default=1)
args = parser.parse_args()

# Custom Params
params = dict()
params['model_folder'] = args.model_folder
params['net_resolution'] = args.net_resolution
params['number_people_max'] = args.number_people_max
params['display'] = 0
params['disable_multi_thread'] = True

# Start openpose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Start webcam
stream = cv2.VideoCapture(0)
stream.set(3, args.cam_width)
stream.set(4, args.cam_height)

frames = 0
start = time.time()
while True:

    # Get image
    frames += 1
    ret, img = stream.read()
    if img is None:
        continue

    # Label image
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.waitAndEmplace([datum])
    opWrapper.waitAndPop([datum])

    # Show image
    if type(datum.poseKeypoints) == np.ndarray and \
       datum.poseKeypoints.size > 0:
        cv2.imshow("", datum.cvOutputData)

    # Check for quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # Print frame rate
    if time.time() - start >= 1:
        print(frames)
        frames = 0
        start = time.time()

stream.release()
cv2.destroyAllWindows()
