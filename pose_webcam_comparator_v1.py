import tensorflow as tf
import numpy as np
import cv2
import time
import argparse
from sklearn.metrics.pairwise import cosine_similarity

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str)
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()


def scale_transform(coords):
    """
    Parameters:
    coords (np.array): Nx2 array of (x,y) coordinates

    Returns:
    np.array: coords scaled to 1x1 with center at (0,0)
    """
    diff = coords.max(axis=0) - coords.min(axis=0)
    diff_max = max(diff)
    out = (coords - coords.mean(axis=0)) / diff_max

    return out


def main():

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        # Get coordinates from target image
        input_image, draw_image, output_scale = posenet.read_imgfile(
                args.image, scale_factor=args.scale_factor, output_stride=output_stride)

        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
            model_outputs,
            feed_dict={'image:0': input_image}
        )

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmaps_result.squeeze(axis=0),
            offsets_result.squeeze(axis=0),
            displacement_fwd_result.squeeze(axis=0),
            displacement_bwd_result.squeeze(axis=0),
            output_stride=output_stride,
            max_pose_detections=1,
            min_pose_score=0.25)

        # Scale, transform, normalize, reshape target image
        target_coords = scale_transform(keypoint_coords[0])
        target_coords /= np.linalg.norm(target_coords)
        target_coords_vec = np.reshape(target_coords, (1, target_coords.shape[0] * 2))

        # Start webcam
        cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0
        while True:
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

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

            if pose_scores[0] == 0:
                continue

            # Scale, transform, normalize, reshape
            coords = scale_transform(keypoint_coords[0])
            coords /= np.linalg.norm(coords)
            coords_vec = np.reshape(coords, (1, coords.shape[0] * 2))

            # Compute similarity score
            cosine_score = cosine_similarity(target_coords_vec, coords_vec)[0]
            position = (10, 50)
            display_image = cv2.putText(display_image,
                                        "{}".format(cosine_score, 5),
                                        position,
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (0, 255, 0),
                                        2)

            keypoint_coords *= output_scale

            # TODO this isn't particularly fast, use GL for drawing and display someday...
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            cv2.imshow('posenet', overlay_image)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    main()
