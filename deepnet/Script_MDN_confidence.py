from poseConfig import aliceConfig as conf
import numpy as np
import os
import PoseUNet_resnet
from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt

conf.cachedir += '_moreeval'
conf.use_unet_loss = True
conf.pretrained_weights = '/groups/branson/bransonlab/mayank/PoseTF/data/pretrained/resnet_tf_v2/20180601_resnet_v2_imagenet_checkpoint/model.ckpt-258931'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
self = PoseUNet_resnet.PoseUMDN_resnet(conf,'joint_deconv_dist_pred')
self.pred_dist = True
V = self.classify_val()

pred_conf = (V[5][-1][:, 40, :]) * self.offset
dist_u_mdn = np.sqrt(np.sum((V[3] - V[-1][1]) ** 2, axis=-1))
dist_combined = np.maximum(pred_conf, dist_u_mdn)
preds = V[3]*self.offset
actual_dist = V[0]

f,ax = plt.subplots(1,2)
ax = ax.flatten()
for ndx in range(2):
    pt = 13 + ndx
    tr = 4
    prm,rcm,_ = precision_recall_curve(actual_dist[:,pt] > tr, pred_conf[:, pt])
    pru,rcu,_ = precision_recall_curve(actual_dist[:,pt] > tr, dist_u_mdn[:, pt])
    prf,rcf,_ = precision_recall_curve(actual_dist[:,pt] > tr, dist_combined[:, pt])
    ax[ndx].plot(rcm,prm)
    ax[ndx].plot(rcu,pru)
    ax[ndx].plot(rcf,prf)
    ax[ndx].legend(['mdn','unet','comb'])
    ax[ndx].set_title('{}'.format(pt))

