"""Tests that Grone's HRNet backbone (mdn_backbone 'hrnet') starts from the ImageNet HRNet-W32 checkpoint.

With mmpose 1.x the checkpoint is given as init_cfg, which mmpose only applies when init_weights() is
called; without that call the backbone silently starts from random weights.

  python test_grone_hrnet_pretrained.py      (or: pytest test_grone_hrnet_pretrained.py)
"""
import torch

from Pose_multi_mdn_joint_torch import my_hrnet_fpn_backbone

HRNET_W32_IMAGENET_URL = 'https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth'


def test_hrnet_backbone_loads_imagenet_weights():
  """Every backbone parameter in the checkpoint must equal the checkpoint's value."""
  checkpoint = torch.hub.load_state_dict_from_url(HRNET_W32_IMAGENET_URL, map_location='cpu')
  checkpoint = checkpoint.get('state_dict', checkpoint)
  backbone_state = my_hrnet_fpn_backbone().backbone.state_dict()
  shared_names = [name for name in backbone_state if name in checkpoint]
  assert len(shared_names) > 100, f'only {len(shared_names)} backbone parameters found in the checkpoint'
  differing_names = [name for name in shared_names if not torch.equal(backbone_state[name], checkpoint[name])]
  assert not differing_names, \
    f'{len(differing_names)} of {len(shared_names)} parameters differ from the ImageNet checkpoint, e.g. {differing_names[:3]}'


if __name__ == '__main__':
  test_hrnet_backbone_loads_imagenet_weights()
  print('all passed')
