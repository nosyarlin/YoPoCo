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

# Start webcam
webcam = get_webcam(args.cam_width, args.cam_height)

# Read target video
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
    img = get_image(webcam, args.cam_width, args.cam_height)
    target_img = get_image(webcam, args.cam_width, args.cam_height)
    if img is None or target_img is None:
        continue

    # Label images
    img_datum = label_img(opWrapper, img)
    target_datum = label_img(opWrapper, target_img)

    # Check if OpenPose managed to label
    if type(img_datum.poseKeypoints) != np.ndarray or \
       img_datum.poseKeypoints.shape != (1, 25, 3):
        continue

    elif type(target_datum.poseKeypoints) != np.ndarray or \
         target_datum.poseKeypoints.shape != (1, 25, 3):
        continue

    # Scale, transform, normalize, reshape, predict
    coords_vec = make_vector(img_datum.poseKeypoints)
    target_coords_vec = make_vector(target_datum.poseKeypoints)
    input_vec = np.concatenate([coords_vec, target_coords_vec]).flatten()
    similarity_score = model.predict(input_vec.reshape((1, -1)))
    ordinal_score = get_ordinal_score(similarity_score)

    # Concatenate webcam and target video
    numpy_horizontal_concat = np.concatenate((img_datum.cvOutputData,
                                              target_datum.cvOutputData),
                                              axis=1)

    # Add overlay to show results
    overlay = numpy_horizontal_concat.copy()
    cv2.rectangle(overlay, (0, 0), (args.cam_width // 2, args.cam_height),
                  get_ordinal_score(similarity_score)[2], -1)
    numpy_horizontal_concat = cv2.addWeighted(overlay, ordinal_score[1],
                                              numpy_horizontal_concat,
                                              1 - ordinal_score[1], 0,
                                              numpy_horizontal_concat)

    # Draw a vertical white line with thickness of 10 px
    cv2.line(numpy_horizontal_concat, (args.cam_width // 2, 0),
             (args.cam_width // 2, args.cam_height),
             (255, 255, 255), 10)

    # Display comment
    cv2.rectangle(numpy_horizontal_concat, (10, 30), (600, 120), (255, 255, 255), 3)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(numpy_horizontal_concat, ' ' + ordinal_score[0], (10, 100), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    # Record Video
    out.write(numpy_horizontal_concat)

    # Display img
    cv2.imshow("Webcam and Target Image", numpy_horizontal_concat)

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
stream.release()
stream_target.release()
out.release()
cv2.destroyAllWindows()
