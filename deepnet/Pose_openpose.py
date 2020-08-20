import open_pose4 as op


class Pose_openpose(object):
    name = 'deepnet'
    conf = None

    def __init__(self,conf,name='deepnet',**kwargs):
        conf.op_backbone = 'resnet50_8px'
        conf.op_backbone_weights = 'imagenet'
        conf.op_map_lores_blur_rad = 1.0
        conf.op_map_hires_blur_rad = 2.0
        conf.op_paf_lores_tubewidth = 0.95 # not used if tubeblur=True
        conf.op_paf_lores_tubeblur = False
        conf.op_paf_lores_tubeblursig = 0.95
        conf.op_paf_lores_tubeblurclip = 0.05
        conf.op_paf_nstage = 5
        conf.op_map_nstage = 1
        conf.op_hires = True
        conf.op_hires_ndeconv = 2
        conf.op_base_lr = 4e-5  # Gines 5e-5
        conf.op_weight_decay_kernel = 5e-4
        conf.op_hmpp_floor = 0.1
        conf.op_hmpp_nclustermax = 1
        conf.op_pred_raw = False
        conf.n_steps = 4.41

        self.name = name
        op.update_conf(conf)
        conf.is_multi = False
        self.conf = conf

    def train_wrapper(self, restore=False):
        op.training(self.conf, self.name)

    def get_pred_fn(self,model_file=None):
        return op.get_pred_fn(self.conf,model_file=model_file,name=self.name)