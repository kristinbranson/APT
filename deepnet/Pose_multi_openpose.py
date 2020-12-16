import Pose_openpose
import open_pose4 as op

class Pose_multi_openpose(Pose_openpose.Pose_openpose):
    def __init__(self,conf,name='deepnet',**kwargs):
        super(Pose_multi_openpose,self).__init__(conf,name=name,**kwargs)
        self.conf.is_multi = True

    def get_pred_fn(self,model_file=None,max_n=None,imsz=None):
        if max_n is not None:
            self.conf.max_n_animals = max_n
        if imsz is not None:
            self.conf.imsz = imsz
        op.update_conf(self.conf)
        return op.get_pred_fn(self.conf,model_file=model_file,name=self.name)