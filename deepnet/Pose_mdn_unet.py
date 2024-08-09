import APT_interface as apt
import PoseUNet_resnet as PoseURes
import tensorflow as tf

class Pose_mdn_unet(PoseURes.PoseUMDN_resnet):
    def __init__(self,conf,**kwargs):
        conf.mdn_use_unet_loss = True
        PoseURes.PoseUMDN_resnet.__init__(self,conf,**kwargs)
        if self.name == 'deepnet':
            self.train_data_name = 'traindata'
        else:
            self.train_data_name = None

    def train_wrapper(self,restore=False,model_file=None):
        PoseURes.PoseUMDN_resnet.train_umdn(self,restore=restore,model_file=model_file)
