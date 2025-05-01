import mmdet
from packaging import version

if version.parse(mmdet.__version__).major >2:
    from Pose_detect_mmdetect3x import Pose_detect_mmdetect
else:
    from Pose_detect_mmdetect2x import Pose_detect_mmdetect
