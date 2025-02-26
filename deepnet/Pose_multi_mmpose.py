import mmpose
from packaging import version

if version.parse(mmpose.__version__).major > 0:
    from Pose_multi_mmpose_new import Pose_multi_mmpose_new as Pose_multi_mmpose
else:
    from Pose_multi_mmpose_old import Pose_multi_mmpose
    