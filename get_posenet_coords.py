import tensorflow as tf
import argparse
import os

import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--image_dir', type=str, default='./data')
parser.add_argument('--output_dir', type=str, default='./coords')
args = parser.parse_args()


def main():

    with tf.Session() as sess:
        # Load model
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        # Make output dir
        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        # Read filepaths for images
        classes = {}
        for f in os.scandir(args.image_dir):
            if f.is_dir():
                filenames = [g.path for g in os.scandir(f) if g.is_file() and g.path.endswith(('.png', '.jpg'))]
                classes[f.name] = filenames

        # Make headers for output csv
        headers = ['score']
        score_headers = []
        coord_headers = []
        for name in posenet.PART_NAMES:
            score_headers.append("{}_score".format(name))
            coord_headers.append("{}_y".format(name))
            coord_headers.append("{}_x".format(name))

        headers = headers + score_headers + coord_headers

        # Get coordinates and scores for all images
        for c in classes:

            # Prepare file name
            filename = os.path.join(args.output_dir, c)
            out = open(filename, "w+")
            out.write(','.join(headers))
            out.write('\n')

            for f in classes[c]:
                # Label image
                input_image, draw_image, output_scale = posenet.read_imgfile(
                    f, scale_factor=args.scale_factor, output_stride=output_stride)

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

                keypoint_coords *= output_scale

                pose_scores.tofile(out, sep=",")
                out.write(",")
                keypoint_scores.tofile(out, sep=",")
                out.write(",")
                keypoint_coords.tofile(out, sep=",")
                out.write("\n")
            out.close()


if __name__ == "__main__":
    main()
