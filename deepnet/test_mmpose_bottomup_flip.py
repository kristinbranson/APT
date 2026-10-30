"""Tests that the bottom-up mmpose training augmentation (DEKR, CiD) keeps keypoints on the image content
when it flips.

mmpose's RandomFlip mirrors keypoints about results['input_size'] if present, unpacking it as (h, w),
but BottomupRandomAffine stores input_size as (w, h). With a non-square network input (APT's conf.imsz),
flipping after the affine puts every keypoint |w - h| pixels away from its image content.

  python test_mmpose_bottomup_flip.py      (or: pytest test_mmpose_bottomup_flip.py)
"""
import tempfile

import numpy as np
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope

import poseConfig
from Pose_mmpose_new import create_mmpose_cfg

DEKR_CONFIG_FILE = 'configs/body_2d_keypoint/dekr/coco/dekr_hrnet-w32_8xb10-140e_coco-512x512.py'
CID_CONFIG_FILE = 'configs/body_2d_keypoint/cid/coco/cid_hrnet-w32_8xb20-140e_coco-512x512.py'


def make_conf(mmpose_net, imsz):
  """Returns a minimal multi-animal conf for a bottom-up mmpose net with network input imsz (h, w)."""
  conf = poseConfig.config()
  conf.cachedir = tempfile.mkdtemp()
  conf.is_multi = True
  conf.mmpose_net = mmpose_net
  conf.imsz = imsz
  conf.rescale = 1
  conf.n_classes = 2
  conf.flipLandmarkMatches = {}
  conf.op_affinity_graph = [[0, 1]]
  conf.horz_flip = True
  conf.vert_flip = False
  conf.rrange = 0
  conf.trange = 0
  conf.scale_factor_range = 1
  conf.max_n_animals = 2
  conf.multi_loss_mask = False
  return conf


def augment_with_flip(mmpose_config_file, mmpose_net, imsz, image_wh):
  """Runs APT's train-time augmentation, with flipping forced on, on a black image_wh image holding two white
  squares, each with one keypoint at its centre. Returns (augmented image, augmented keypoints)."""
  init_default_scope('mmpose')
  cfg = create_mmpose_cfg(make_conf(mmpose_net, imsz), mmpose_config_file, 'test')
  stages = []
  for stage in cfg.train_dataloader.dataset.pipeline:
    if stage['type'] in ['LoadImage', 'GenerateTarget', 'BottomupGetHeatmapMask', 'PackPoseInputs']:
      continue
    stage = dict(stage)
    if stage['type'] == 'RandomFlip':
      stage['prob'] = 1.
    stages.append(stage)
  image_width, image_height = image_wh
  img = np.zeros((image_height, image_width, 3), np.uint8)
  keypoints = np.array([[[100., 150.], [image_width - 120., image_height - 90.]]], np.float32)
  for x, y in keypoints[0]:
    img[int(y) - 20:int(y) + 20, int(x) - 20:int(x) + 20] = 255
  results = dict(img=img, img_shape=img.shape[:2], ori_shape=img.shape[:2],
                 keypoints=keypoints, keypoints_visible=np.ones((1, 2), np.float32),
                 bbox=np.array([[50., 50., image_width - 50., image_height - 50.]], np.float32),
                 area=np.array([1.], np.float32), flip_indices=[0, 1], id=0, img_id=0)
  out = Compose(stages)(results)
  assert out['flip'], 'flip was forced on'
  return out['img'], out['keypoints'][0]


def assert_keypoints_on_white_squares(img, keypoints):
  """Each keypoint must still sit on (a white pixel of) its square after augmentation."""
  for x, y in keypoints:
    xi, yi = int(round(x)), int(round(y))
    is_inside = 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]
    assert is_inside and img[yi, xi].min() > 200, \
      f'keypoint ({x:.0f}, {y:.0f}) is not on its square after flipping (image {img.shape[1]}x{img.shape[0]})'


def test_dekr_flip_keeps_keypoints_on_image_content_non_square_input():
  """DEKR, network input 1280x736 (non-square), 720x576 frames: flipped keypoints stay on their content."""
  img, keypoints = augment_with_flip(DEKR_CONFIG_FILE, 'dekr', (736, 1280), (720, 576))
  assert_keypoints_on_white_squares(img, keypoints)


def test_dekr_flip_keeps_keypoints_on_image_content_same_size_frames():
  """DEKR, frames the same size as the non-square network input: still must stay aligned."""
  img, keypoints = augment_with_flip(DEKR_CONFIG_FILE, 'dekr', (736, 1280), (1280, 736))
  assert_keypoints_on_white_squares(img, keypoints)


def test_cid_flip_keeps_keypoints_on_image_content_non_square_input():
  """CiD uses the same bottom-up pipeline, so it must stay aligned too."""
  img, keypoints = augment_with_flip(CID_CONFIG_FILE, 'cid', (736, 1280), (720, 576))
  assert_keypoints_on_white_squares(img, keypoints)


def test_dekr_flip_keeps_keypoints_on_image_content_square_input():
  """Square network input: aligned before and after the fix (the bug needs w != h)."""
  img, keypoints = augment_with_flip(DEKR_CONFIG_FILE, 'dekr', (512, 512), (720, 576))
  assert_keypoints_on_white_squares(img, keypoints)


if __name__ == '__main__':
  for name, fn in list(globals().items()):
    if name.startswith('test_'):
      fn()
  print('all passed')
