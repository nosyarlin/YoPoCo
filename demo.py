import sys
import cv2
import argparse
import numpy as np
import time
from keras.models import load_model

sys.path.append('/usr/local/python')

from openpose import pyopenpose as op

# Flags
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='../openpose/models/')
parser.add_argument('--target_video', type=str, default='./yoga.mp4')
parser.add_argument('--net_resolution', type=str, default='176x176')
parser.add_argument('--cam_width', type=int, default=1920)
parser.add_argument('--cam_height', type=int, default=1080)
parser.add_argument('--display_width', type=int, default=1920)
parser.add_argument('--display_height', type=int, default=1080)
parser.add_argument('--number_people_max', type=int, default=1)

args = parser.parse_args()

# Custom Params
params = dict()
params['model_folder'] = args.model_folder
params['net_resolution'] = args.net_resolution
params['number_people_max'] = args.number_people_max
params['display'] = 0
params['disable_multi_thread'] = True


def scale_transform(coords):
    """
    Parameters:
    coords (25x3 ndarray): array of (x,y,c) coordinates

    Returns:
    ndarray: coords scaled to 1x1 with center at (0,0)
    ndarray: confidence scores of each joint
    """
    coords, scores = coords[:, :, :-1], coords[:, :, -1]
    diff = coords.max(axis=1) - coords.min(axis=1)
    diff_max = np.max(diff, axis=0)
    mean = coords.mean(axis=1).reshape(coords.shape[0], 1, coords.shape[-1])
    out = (coords - mean) / diff_max

    return out, scores


def make_vector(poseKeypoints):
    """
    Parameters:
    poseKeypoints (ndarray): Single person output from OpenPose

    Returns:
    ndarray: scaled, transformed, normalized row vector
    """
    N, D, C = poseKeypoints.shape
    coords, pose_scores = scale_transform(poseKeypoints)
    pose_scores = pose_scores.reshape((N, D, 1))
    coords_vec = np.concatenate([coords, pose_scores], axis=2)
    coords_vec /= np.linalg.norm(coords_vec, axis=2)[:, :, np.newaxis]

    return coords_vec


def get_ordinal_score(score):
    """
    Parameters:
    score (float): similarity score between two poses
                   between 0 and 1

    Returns:
    string: string text of the results
    float: transparency value
    tuple: color of overlay
    """
    alpha = 0.2
    overlay_color = (0, 0, 255)

    if score > 0.712:
        out = "Genius!"
        overlay_color = (0, 255, 0)
    elif score > 0.459:
        out = "Almost there!"
        overlay_color = (255, 150, 0)
    elif score > 0.298:
        out = "Nice try!"
    else:
        out = "Try harder!"

    return out, alpha, overlay_color


def cropped_image(full_image):
    w_min = 960 - (args.display_width // 4)
    w_max = 960 + (args.display_width // 4)
    out = full_image[0:args.display_height, w_min:w_max]
    return out


# Start openpose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Start webcam
stream = cv2.VideoCapture(0)
if (stream.isOpened() is False):
    print("Error opening video stream or file")
    raise SystemExit(1)
stream.set(3, args.cam_width)
stream.set(4, args.cam_height)

# Read target video
stream_target = cv2.VideoCapture(args.target_video)
stream_target_window = np.zeros((args.cam_height, round(args.cam_width / 2), 3), dtype=np.uint8)  # has to be same edge dimension window

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('yoga_output.avi', fourcc, 10.0, (args.display_width, args.display_height))

# Setup framerate params
frames = 0
framerate = 0
start = time.time()
time.sleep(2)  # delay to wait for detection
model = load_model('ComparatorNet.h5')

while True:
    # Get image from webcam
    frames += 1
    ret, img_original = stream.read()
    img = cv2.flip(cropped_image(img_original), 1)

    # Get image from target video
    target_ret, stream_target_img_original = stream_target.read()
    stream_target_img = cv2.flip(cropped_image(stream_target_img_original), 1)
    if img is None:
        continue

    # Label image
    datum = op.Datum()
    datum.cvInputData = img
    opWrapper.waitAndEmplace([datum])
    opWrapper.waitAndPop([datum])

    # Label target image
    datum_target = op.Datum()
    datum_target.cvInputData = stream_target_img
    opWrapper.waitAndEmplace([datum_target])
    opWrapper.waitAndPop([datum_target])

    # Check if OpenPose managed to label
    if type(datum.poseKeypoints) != np.ndarray or \
       datum.poseKeypoints.shape != (1, 25, 3):
        continue

    elif type(datum_target.poseKeypoints) != np.ndarray or \
         datum_target.poseKeypoints.shape != (1, 25, 3):
        continue

    # Scale, transform, normalize, reshape, predict
    coords_vec = make_vector(datum.poseKeypoints)
    target_coords_vec = make_vector(datum_target.poseKeypoints)
    input_vec = np.concatenate([coords_vec, target_coords_vec]).flatten()
    similarity_score = model.predict(input_vec.reshape((1, -1)))
    ordinal_score = get_ordinal_score(similarity_score)

    # Fit target video into stream target window
    stream_target_window[0:0 + datum_target.cvOutputData.shape[0],
    0:0 + datum_target.cvOutputData.shape[1]] = datum_target.cvOutputData

    # Concatenate webcam and target video
    numpy_horizontal_concat = np.concatenate((datum.cvOutputData,
                                              stream_target_window),
                                              axis=1)

    # Add overlay to show results
    overlay = numpy_horizontal_concat.copy()
    cv2.rectangle(overlay, (0, 0), (args.display_width // 2, args.display_height),
                  get_ordinal_score(similarity_score)[2], -1)
    numpy_horizontal_concat = cv2.addWeighted(overlay, ordinal_score[1],
                                              numpy_horizontal_concat,
                                              1 - ordinal_score[1], 0,
                                              numpy_horizontal_concat)

    # Draw a vertical white line with thickness of 10 px
    cv2.line(numpy_horizontal_concat, (args.display_width // 2, 0),
             (args.display_width // 2, args.display_height),
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
