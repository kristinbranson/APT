#! /usr/bin/env python3
"""Smoke-test a pose-estimation environment/image end to end.

Runs the mmpose top-down demo on a fixed input image and compares the rendered
result to a known-good target image, passing/failing accordingly.  This
consolidates what used to be three scripts: image_demo.py (the computation),
test.py (the orchestration), and compare_to_target.py (the comparison).

The comparison is done in Python with numpy (a normalized RMS pixel
difference), rather than by shelling out to ImageMagick, which suits a
scientific-computing environment and lets the whole test -- compute and compare
-- run inside the environment or image under test.

Run it inside the environment/image under test, e.g.:

    conda run --name <env> python test_pose_estimation.py

or inside a container.  A working GPU is required.  Relies on numpy and the mm*
stack (mmcv/mmpose/mmengine) that the environment under test provides.
"""

import argparse
import contextlib
import logging
import os
import sys
import tempfile
import urllib.request
import warnings

# mmengine imports pkg_resources, which emits a deprecation UserWarning on
# import.  Silence just that warning, before importing the mm* packages that
# trigger it.
warnings.filterwarnings(
    'ignore',
    message='pkg_resources is deprecated as an API',
    category=UserWarning)

import numpy as np
from mmcv.image import imread
from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


# The fixed smoke-test assets (all live in this script's directory).
DEMO_INPUT_IMAGE = 'demo-input.jpg'
DEMO_CONFIG = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
DEMO_CHECKPOINT = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
DEFAULT_OUTPUT_IMAGE = 'demo-output.jpg'
DEFAULT_TARGET_IMAGE = 'demo-output-target.jpg'

# Visualization settings, matching the defaults the target image was rendered
# with.
KEYPOINT_THRESHOLD = 0.3
KEYPOINT_RADIUS = 3
LINE_THICKNESS = 1
BOX_ALPHA = 0.8
SKELETON_STYLE = 'mmpose'

# Maximum normalized RMS pixel difference (over [0, 1]) for the output to count
# as matching the target.  A JPEG re-encode of the same render scores under
# 0.02, while a small shift or a genuinely different render scores 0.15 or more,
# so 0.05 leaves a wide margin on both sides.
DEFAULT_THRESHOLD = 0.05

# Known download URL(s) for the checkpoint, keyed by file basename.  Used to
# fetch the checkpoint from the OpenMMLab model zoo when it is not present.
CHECKPOINT_URL_BY_BASENAME = {
    'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth':
        'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/'
        'topdown_heatmap/coco/'
        'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth',
}


def ensure_checkpoint(checkpointPath):
  # Download the checkpoint to checkpointPath if it is not already present.
  if os.path.exists(checkpointPath):
    return
  basename = os.path.basename(checkpointPath)
  url = CHECKPOINT_URL_BY_BASENAME.get(basename)
  if url is None:
    raise FileNotFoundError(
        'Checkpoint %s does not exist and no download URL is known for it.' % checkpointPath)
  print_log('Checkpoint %s not found; downloading from %s' % (checkpointPath, url),
            logger='current', level=logging.INFO)
  # Download to a temporary file in the same directory, then rename, so an
  # interrupted download does not leave a truncated file at the final path.
  destinationDirectory = os.path.dirname(os.path.abspath(checkpointPath))
  os.makedirs(destinationDirectory, exist_ok=True)
  fileDescriptor, temporaryPath = tempfile.mkstemp(dir=destinationDirectory, suffix='.partial')
  os.close(fileDescriptor)
  try:
    urllib.request.urlretrieve(url, temporaryPath)
    os.replace(temporaryPath, checkpointPath)
  except BaseException:
    if os.path.exists(temporaryPath):
      os.remove(temporaryPath)
    raise


def run_pose_estimation(inputImagePath, configPath, checkpointPath, outputImagePath, device):
  # Run the top-down pose-estimation demo and write the rendered result (with
  # heatmap) to outputImagePath.
  ensure_checkpoint(checkpointPath)

  # Ask the model to also output heatmaps, so they can be drawn.
  configOverrides = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

  # init_model prints a bare "Loads checkpoint ..." line to stdout via print()
  # (not the logger), so redirect stdout to silence just that.
  with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
    model = init_model(configPath, checkpointPath, device=device, cfg_options=configOverrides)

  model.cfg.visualizer.radius = KEYPOINT_RADIUS
  model.cfg.visualizer.alpha = BOX_ALPHA
  model.cfg.visualizer.line_width = LINE_THICKNESS
  visualizer = VISUALIZERS.build(model.cfg.visualizer)
  visualizer.set_dataset_meta(model.dataset_meta, skeleton_style=SKELETON_STYLE)

  batchResults = inference_topdown(model, inputImagePath)
  results = merge_data_samples(batchResults)

  inputImage = imread(inputImagePath, channel_order='rgb')
  visualizer.add_datasample(
      'result',
      inputImage,
      data_sample=results,
      draw_gt=False,
      draw_bbox=True,
      kpt_thr=KEYPOINT_THRESHOLD,
      draw_heatmap=True,
      show_kpt_idx=False,
      skeleton_style=SKELETON_STYLE,
      show=False,
      out_file=outputImagePath)


def normalized_rms_difference(outputImagePath, targetImagePath):
  # Compare two images by normalized RMS pixel difference in [0, 1].  Return
  # (matchesDimensions, distance); distance is None when the dimensions differ.
  outputImage = imread(outputImagePath, channel_order='rgb').astype(np.float64)
  targetImage = imread(targetImagePath, channel_order='rgb').astype(np.float64)
  if outputImage.shape != targetImage.shape:
    return False, None
  distance = float(np.sqrt(np.mean((outputImage - targetImage) ** 2)) / 255.0)
  return True, distance


def main():
  # Parse arguments, run the demo, and compare its output to the target.
  argumentParser = argparse.ArgumentParser(description='Run the pose-estimation smoke test.')
  argumentParser.add_argument(
      '--device', default='cuda:0', help='Device used for inference (default: cuda:0)')
  argumentParser.add_argument(
      '--output', default=DEFAULT_OUTPUT_IMAGE, help='Where to write the rendered output image')
  argumentParser.add_argument(
      '--target', default=DEFAULT_TARGET_IMAGE, help='Known-good target image to compare against')
  argumentParser.add_argument(
      '--threshold', type=float, default=DEFAULT_THRESHOLD,
      help='Maximum normalized RMS pixel difference to count as a match')
  arguments = argumentParser.parse_args()

  # Resolve asset paths relative to this script's directory, so the test works
  # regardless of the current working directory.
  scriptDirectory = os.path.dirname(os.path.abspath(__file__))
  os.chdir(scriptDirectory)

  # Run the computation.  Its (noisy) progress goes to stderr; the final
  # pass/fail line goes to stdout, so the verdict is not interleaved with it.
  run_pose_estimation(DEMO_INPUT_IMAGE, DEMO_CONFIG, DEMO_CHECKPOINT, arguments.output, arguments.device)

  # Compare the rendered output to the target.
  if not os.path.isfile(arguments.target):
    print('FAIL: target image %s does not exist' % arguments.target)
    sys.exit(1)
  matchesDimensions, distance = normalized_rms_difference(arguments.output, arguments.target)
  if not matchesDimensions:
    print('FAIL: %s and %s have different dimensions' % (arguments.output, arguments.target))
    sys.exit(1)
  if distance <= arguments.threshold:
    print('PASS: %s matches %s (normalized RMS difference %.4f <= %.4f)'
          % (arguments.output, arguments.target, distance, arguments.threshold))
  else:
    print('FAIL: %s differs from %s (normalized RMS difference %.4f > %.4f)'
          % (arguments.output, arguments.target, distance, arguments.threshold))
    sys.exit(1)


if __name__ == '__main__':
  main()
