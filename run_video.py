# # TransNet: A deep network for fast detection of common shot transitions
# This repository contains code for paper *TransNet: A deep network for fast detection of common shot transitions*.
#
# If you use it in your work, please cite:
#
#
#     @article{soucek2019transnet,
#         title={TransNet: A deep network for fast detection of common shot transitions},
#         author={Sou{\v{c}}ek, Tom{\'a}{\v{s}} and Moravec, Jaroslav and Loko{\v{c}}, Jakub},
#         journal={arXiv preprint arXiv:1906.03363},
#         year={2019}
#     }

# ## How to use it?
#
# Firstly, *tensorflow* needs to be installed.
# Do so by doing:
#
#     pip install tensorflow
#
# If you want to run **TransNet** directly on video files, *ffmpeg* needs to be installed as well:
#
#     pip install ffmpeg-python
#
# You can also install *pillow* for visualization:
#
#     pip install pillow
#
#
# Tested with *tensorflow* v1.12.0.


import argparse
import logging
import random
from pathlib import Path

import ffmpeg
import numpy as np
import tensorflow as tf
from script_utils.common import common_setup
from tqdm import tqdm

from transnet import TransNetParams, TransNet
from transnet_utils import scenes_from_predictions


def process(net, params, video_path):
    # export video into numpy array using ffmpeg
    video_stream, err = (ffmpeg.input(video_path).output(
        'pipe:',
        format='rawvideo',
        pix_fmt='rgb24',
        s='{}x{}'.format(params.INPUT_WIDTH,
                         params.INPUT_HEIGHT)).run(capture_stdout=True))
    video = np.frombuffer(video_stream, np.uint8).reshape(
        [-1, params.INPUT_HEIGHT, params.INPUT_WIDTH, 3])

    # predict transitions using the neural network
    return net.predict_video(video)


def main():
    # Use first line of file docstring as description if it exists.
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0] if __doc__ else '',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('videos_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--thresh', default=0.1, type=float)
    parser.add_argument('--extensions', nargs='+', default=['.mp4', '.mkv'])

    args = parser.parse_args()
    args.extensions = [
        x if x[0] == '.' else ('.' + x) for x in args.extensions
    ]

    args.output_dir.mkdir(exist_ok=True, parents=True)
    common_setup(__file__, args.output_dir, args)
    videos = [
        x for ext in args.extensions for x in args.videos_dir.rglob('*' + ext)
    ]
    logging.info(f'Processing {len(videos)} videos.')
    random.shuffle(videos)

    # initialize the network
    params = TransNetParams()
    params.CHECKPOINT_PATH = "./model/transnet_model-F16_L3_S2_D256"

    net = TransNet(params)

    for video_path in videos:
        output_dir = args.output_dir / (video_path.relative_to(
            args.videos_dir).with_suffix(''))
        output_dir.mkdir(exist_ok=True, parents=True)
        output_predictions = output_dir / 'predictions.npy'
        output_scenes = output_dir / 'scenes.txt'

        if output_predictions.exists() and output_scenes.exists():
            logging.info(f'{output_dir} already processed, skipping.')
            continue
        predictions = process(net, params, video_path)
        scenes = scenes_from_predictions(predictions, threshold=args.thresh)

        np.save(output_predictions, predictions)
        np.savetxt(output_scenes, scenes, fmt='%d')


if __name__ == "__main__":
    main()
