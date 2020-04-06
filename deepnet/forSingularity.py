
#
#
#from poseConfig import aliceConfig as conf
#
#import PoseUNet
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#conf.batch_size = 16
#name = 'pose_unet_increasing_filter_32_bsize16'
## self = PoseUNet.PoseUNet(conf,name=name,for_training=False)
## val_dist, val_ims, val_preds, val_predlocs, val_locs, val_info = self.classify_val(0)
#self = PoseUNet.PoseUNet(conf,name=name)
#self.train_unet(False,0)
#
##
# from stephenHeadConfig import conf
# import tensorflow as tf
#
# conf.scale_range = 0.1
#
# import PoseUNet
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
# name = 'pose_unet_increasing_filter_32'
# self = PoseUNet.PoseUNet(conf,name=name)
# self.train_unet(False,0)
#
# tf.reset_default_graph()
# self = PoseUNet.PoseUNet(conf,name=name)
# val_dist, val_ims, val_preds, val_predlocs, val_locs, val_info = self.classify_val(0)



#
# import PoseUNet
# from stephenHeadConfig import sideconf as conf
# import tensorflow as tf
# import multiResData
#
# self = PoseUNet.PoseUNet(conf,name='pose_unet_full_20180601')
# self.train_unet(False,train_type=1)
#


## running Alice's tracking.
import PoseUNet
from poseConfig import aliceConfig as conf
import APT_interface
import PoseTools
import tensorflow as tf
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tf.reset_default_graph()
#dirs = ['/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/witinAssayData/cx_GMR_SS00006_CsChr_RigC_20151014T093157',
#'/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/witinAssayData/cx_GMR_SS00217_CsChr_RigB_20150929T095409',
#'/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/witinAssayData/cx_GMR_SS00277_CsChr_RigD_20150819T094557',
#'/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/witinAssayData/cx_JRC_SS03500_CsChr_RigB_20150811T114536',
#'/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/witinAssayData/cx_GMR_SS00243_CsChr_RigD_20150812T155340'
#'/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/withGenotypeData/cx_GMR_SS00030_CsChr_RigD_20150826T145307'
#'/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/withGenotypeData/cx_GMR_SS00038_CsChr_RigC_20150908T140450'
#'/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/withGenotypeData/cx_GMR_SS00168_CsChr_RigC_20150909T110456',
#'/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/withGenotypeData/cx_GMR_SS00238_CsChr_RigD_20150826T143306',
#'/groups/branson/home/robiea/Projects_data/Labeler_APT/Austin_labelerprojects_expandedbehaviors/GT/withGenotypeData/cx_JHS_K_85321_CsChr_RigB_20151021T095119',
dirs = ['/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00038_CsChr_RigB_20150729T150617',
'/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00030_CsChr_RigC_20150826T144616',
'/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_JHS_K_85321_CsChr_RigD_20150909T163219',
'/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00168_CsChr_RigD_20150909T111218',
'/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00238_CsChr_RigC_20151007T150343',
        ]

self = PoseUNet.PoseUNet(conf, name='pose_unet_full_20180521')
sess = self.init_net_meta(1)
model_file = self.get_latest_model_file()

def pred_fn(all_f):
    bsize = conf.batch_size
    xs, _ = PoseTools.preprocess_ims(
        all_f, in_locs=np.zeros([bsize, self.conf.n_classes, 2]), conf=self.conf,
        distort=False, scale=self.conf.unet_rescale)

    self.fd[self.ph['x']] = xs
    self.fd[self.ph['phase_train']] = False
    self.fd[self.ph['keep_prob']] = 1.
    pred = sess.run(self.pred, self.fd)
    base_locs = PoseTools.get_pred_locs(pred)
    base_locs = base_locs * conf.unet_rescale
    return base_locs

for bdir in dirs:
    mov = os.path.join(bdir,'movie.ufmf')
    trx = os.path.join(bdir,'registered_trx.mat')
    out_file = os.path.join(bdir,'movie_unet_20180512.trk')
    APT_interface.classify_movie(conf,pred_fn, mov, out_file, trx,model_file=model_file)


# import PoseUNetAttention
# import tensorflow as tf
# from poseConfig import aliceConfig as conf
#
# self = PoseUNetAttention.PoseUNetAttention(conf,)
