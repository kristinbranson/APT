# Copyright (c) OpenMMLab. All rights reserved.
import contextlib
import logging
import os
import tempfile
import urllib.request
import warnings
from argparse import ArgumentParser

# mmengine imports pkg_resources, which emits a deprecation UserWarning on
# import.  Silence just that warning, before importing the mm* packages that
# trigger it.
warnings.filterwarnings(
    'ignore',
    message='pkg_resources is deprecated as an API',
    category=UserWarning)

from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


# Known download URLs for checkpoints, keyed by file basename.  Used to fetch a
# checkpoint from the OpenMMLab model zoo when it is not present locally.
CHECKPOINT_URL_BY_BASENAME = {
    'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth':
        'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/'
        'topdown_heatmap/coco/'
        'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth',
}


def ensure_checkpoint(checkpoint):
    # Download the checkpoint to the given path if it does not already exist.
    if os.path.exists(checkpoint):
        return
    basename = os.path.basename(checkpoint)
    url = CHECKPOINT_URL_BY_BASENAME.get(basename)
    if url is None:
        raise FileNotFoundError(
            'Checkpoint {} does not exist and no download URL is known for it.'
            .format(checkpoint))
    print_log(
        'Checkpoint {} not found; downloading from {}'.format(checkpoint, url),
        logger='current',
        level=logging.INFO)
    # Download to a temporary file in the same directory, then rename, so an
    # interrupted download does not leave a truncated file at the final path.
    destination_directory = os.path.dirname(os.path.abspath(checkpoint))
    os.makedirs(destination_directory, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=destination_directory, suffix='.partial')
    os.close(file_descriptor)
    try:
        urllib.request.urlretrieve(url, temporary_path)
        os.replace(temporary_path, checkpoint)
    except BaseException:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        help='Visualize the predicted heatmap')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # download the checkpoint if it is not already present
    ensure_checkpoint(args.checkpoint)

    # build the model from a config file and a checkpoint file
    if args.draw_heatmap:
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    else:
        cfg_options = None

    # init_model prints a bare "Loads checkpoint by ... backend from path: ..."
    # line to stdout via print() (not the logger), so redirect stdout to
    # silence just that.
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        model = init_model(
            args.config,
            args.checkpoint,
            device=args.device,
            cfg_options=cfg_options)

    # init visualizer
    model.cfg.visualizer.radius = args.radius
    model.cfg.visualizer.alpha = args.alpha
    model.cfg.visualizer.line_width = args.thickness

    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style=args.skeleton_style)

    # inference a single image
    batch_results = inference_topdown(model, args.img)
    results = merge_data_samples(batch_results)

    # show the results
    img = imread(args.img, channel_order='rgb')
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results,
        draw_gt=False,
        draw_bbox=True,
        kpt_thr=args.kpt_thr,
        draw_heatmap=args.draw_heatmap,
        show_kpt_idx=args.show_kpt_idx,
        skeleton_style=args.skeleton_style,
        show=args.show,
        out_file=args.out_file)


if __name__ == '__main__':
    main()
