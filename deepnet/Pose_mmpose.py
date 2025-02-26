import mmpose
from packaging import version

if version.parse(mmpose.__version__).major > 0:
    from Pose_mmpose_new import Pose_mmpose
else:
    from Pose_mmpose_old import Pose_mmpose
