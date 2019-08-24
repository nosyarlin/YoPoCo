import sys
import cv2
import argparse
import numpy as np
import time
from keras.models import load_model
from util import (get_ordinal_score, make_vector, get_webcam, get_image, label_img)

sys.path.append('/usr/local/python')

from openpose import pyopenpose as op

# Flags
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='../openpose/models/')
parser.add_argument('--target_video', type=str, default='./yoga.mp4')
parser.add_argument('--net_resolution', type=str, default='176x176')
parser.add_argument('--cam_width', type=int, default=1920)
parser.add_argument('--cam_height', type=int, default=1080)
parser.add_argument('--number_people_max', type=int, default=1)

args = parser.parse_args()

# Custom openpose params
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

# Start streams
webcam = get_webcam(args.cam_width, args.cam_height)
target = cv2.VideoCapture(args.target_video)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter(
    'yoga_output.avi',
    fourcc,
    10.0,
    (args.cam_width, args.cam_height)
)

# Setup framerate params
frames = 0
framerate = 0
start = time.time()
time.sleep(2)  # delay to wait for detection
model = load_model('ComparatorNet.h5')

while True:
    frames += 1

    # Get images
    webcam_img = get_image(webcam, args.cam_width, args.cam_height)
    target_img = get_image(target, args.cam_width, args.cam_height)
    if webcam_img is None or target_img is None:
        continue

    # Label images
    webcam_datum = label_img(opWrapper, webcam_img)
    target_datum = label_img(opWrapper, target_img)

    # Check if OpenPose managed to label
    ordinal_score = ('', 0.0, (0, 0, 0))
    if type(webcam_datum.poseKeypoints) == np.ndarray and \
       webcam_datum.poseKeypoints.shape == (1, 25, 3):

        if type(target_datum.poseKeypoints) == np.ndarray or \
             target_datum.poseKeypoints.shape == (1, 25, 3):

            # Scale, transform, normalize, reshape, predict
            coords_vec = make_vector(webcam_datum.poseKeypoints)
            target_coords_vec = make_vector(target_datum.poseKeypoints)
            input_vec = np.concatenate([coords_vec, target_coords_vec]).flatten()
            similarity_score = model.predict(input_vec.reshape((1, -1)))
            ordinal_score = get_ordinal_score(similarity_score)

    # Concatenate webcam and target video
    screen_out = np.concatenate((webcam_datum.cvOutputData,
                                 target_datum.cvOutputData),
                                axis=1)

    # Add overlay to show results
    overlay = screen_out.copy()
    cv2.rectangle(overlay, (0, 0), (args.cam_width // 2, args.cam_height),
                  ordinal_score[2], -1)
    screen_out = cv2.addWeighted(overlay, ordinal_score[1],
                                 screen_out,
                                 1 - ordinal_score[1], 0,
                                 screen_out)

    # Draw a vertical white line with thickness of 10 px
    cv2.line(screen_out, (args.cam_width // 2, 0),
             (args.cam_width // 2, args.cam_height),
             (255, 255, 255), 10)

    # Display comment
    cv2.rectangle(screen_out, (10, 30), (600, 120), (255, 255, 255), 3)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(screen_out, ' ' + ordinal_score[0], (10, 100), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    # Record Video
    out.write(screen_out)

    # Display img
    cv2.imshow("Webcam and Target Image", screen_out)

    # Check for quit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    # Print frame rate
    if time.time() - start >= 1:
        framerate = frames
        print('Framerate: ', framerate)
        frames = 0
        start = time.time()

# Clean up
webcam.release()
target.release()
out.release()
cv2.destroyAllWindows()
