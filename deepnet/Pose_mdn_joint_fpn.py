from Pose_mdn_joint import Pose_mdn_joint

class Pose_mdn_joint_fpn(Pose_mdn_joint):

    def __init__(self, conf, **kwargs):
        conf.mdn_joint_use_fpn = True
        Pose_mdn_joint.__init__(self,conf,**kwargs)
