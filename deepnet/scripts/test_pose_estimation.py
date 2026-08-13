#! /usr/bin/env python3
"""Smoke-test a pose-estimation environment/image end to end.

Runs the mmpose top-down demo on a fixed input image and compares the rendered
result to a known-good target image, passing/failing accordingly.  On success
the rendered output image is deleted; on failure it is left for inspection.
This consolidates what used to be three scripts: image_demo.py (the
computation), test.py (the orchestration), and compare_to_target.py (the
comparison).

The comparison is done in Python with scikit-image (single-scale structural
similarity, SSIM), rather than by shelling out to ImageMagick, which suits a
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

from mmcv.image import imread
from mmengine.logging import print_log
from skimage.metrics import structural_similarity

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

# Minimum structural similarity (single-scale SSIM, Wang et al. 2004) for the
# output to count as matching the target.  Identical renders score 1.0, and a
# JPEG re-encode of the same render stays above ~0.92, while a small shift or a
# genuinely different render scores below ~0.65, so 0.90 leaves a wide margin on
# both sides.
DEFAULT_THRESHOLD = 0.90

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


def structural_similarity_score(outputImagePath, targetImagePath):
  # Compare two images by single-scale SSIM (Wang et al. 2004).  Return
  # (matchesDimensions, score); score is None when the dimensions differ.  SSIM
  # is 1.0 for identical images and lower for more-different ones.
  outputImage = imread(outputImagePath, channel_order='rgb')
  targetImage = imread(targetImagePath, channel_order='rgb')
  if outputImage.shape != targetImage.shape:
    return False, None
  score = float(structural_similarity(outputImage, targetImage,
                                      channel_axis=-1, data_range=255,
                                      gaussian_weights=True, sigma=1.5,
                                      use_sample_covariance=False))
  return True, score


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
      help='Minimum SSIM to count as a match')
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
  matchesDimensions, score = structural_similarity_score(arguments.output, arguments.target)
  if not matchesDimensions:
    print('FAIL: %s and %s have different dimensions' % (arguments.output, arguments.target))
    sys.exit(1)
  if score >= arguments.threshold:
    print('PASS: %s matches %s (SSIM %.4f >= %.4f)'
          % (arguments.output, arguments.target, score, arguments.threshold))
    # Clean up the output on success; on failure it is left for inspection.
    os.remove(arguments.output)
  else:
    print('FAIL: %s differs from %s (SSIM %.4f < %.4f)'
          % (arguments.output, arguments.target, score, arguments.threshold))
    sys.exit(1)


if __name__ == '__main__':
  main()
