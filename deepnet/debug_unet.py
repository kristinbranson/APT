
device = '0'

name = ''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device

from stephenHeadConfig import conf as conf

import PoseUMDN
import tensorflow as tf
import PoseTools
import math
import PoseUNet
import cv2

import PoseCommon
mov = '/home/mayank/Dropbox (HHMI)/temp/stephen/C002H001S0001_c_2.avi'
out = '/home/mayank/temp/C002H001S0001_c_2_res_unet.avi'
conf.mdn_min_sigma = 3.
conf.mdn_max_sigma = 3.
self = PoseUNet.PoseUNet(conf,'pose_unet_droput_0p7')
# self = PoseUMDN.PoseUMDN(conf)
self.train_unet(False,0)
# self.train_umdn(False,0,joint=True)
val_dist, val_ims, val_preds, val_predlocs, val_locs= self.classify_val(0,-1)
# self.create_pred_movie(mov,out,flipud=True)


##
tf.reset_default_graph()
self.init_train(0)
self.pred = self.create_network()
self.create_saver()
sess = tf.InteractiveSession()
self.init_and_restore(sess,True,['loss','dist'])
info= []
for step in range(300):
    self.setup_val(sess)
    info.append(self.info)


##

f, ax = plt.subplots(2, 4)
ax = ax.flatten()

for bnum in range(int(sel.size/conf.batch_size)):
    # bnum = 0  # 13 #9
    self.fd[self.ph['phase_train']] = False
    bb = self.conf.batch_size

    sel_im = val_ims[sel[bb*bnum:bb*(bnum+1)],...]
    sel_locs = val_locs[sel[bb*bnum:bb*(bnum+1)],...]
    pred_locs = val_predlocs[sel[bb*bnum:bb*(bnum+1)],...]

    in1 = PoseTools.scale_images(sel_im,2,conf)
    self.fd[self.ph['x']] = in1
    rescale = self.conf.unet_rescale
    imsz = [self.conf.imsz[0] / rescale, self.conf.imsz[1] / rescale, ]
    label_ims = PoseTools.create_label_images(
        sel_locs, imsz, 1, self.conf.label_blur_rad)
    self.fd[self.ph['y']] = label_ims
    b_pred = sess.run(self.pred,self.fd)


    for ndx in range(8):
        ax[ndx].cla()
        ax[ndx].imshow(sel_im[ndx, ..., 0], cmap='gray')
        ax[ndx].scatter(sel_locs[ndx, :, 0] , sel_locs[ndx, :, 1] )
        ax[ndx].scatter(pred_locs[ndx, :, 0] * 2, pred_locs[ndx, :, 1] * 2)

    plt.pause(2)

##

val_file = os.path.join(self.conf.cachedir, self.conf.fulltrainfilename + '.tfrecords')
num_val = 0
for record in tf.python_io.tf_record_iterator(val_file):
    num_val += 1

# self.close_cursors()
self.create_cursors(sess)

train_ims = []
train_locs = []
self.conf.brange = [0,0]
self.conf.crange = [1,1]
for step in range(num_val / self.conf.batch_size):
    self.read_images(self.DBType.Train, True, sess, False)
    rescale = self.conf.unet_rescale
    self.fd[self.ph['x']] = PoseTools.scale_images(
        self.xs, rescale, self.conf)
    imsz = [self.conf.imsz[0] / rescale, self.conf.imsz[1] / rescale, ]
    label_ims = PoseTools.create_label_images(
        self.locs / rescale, imsz, 1, self.conf.label_blur_rad)
    self.fd[self.ph['y']] = label_ims
    train_ims.append(self.xs)
    train_locs.append(self.locs)

def val_reshape(in_a):
    in_a = np.array(in_a)
    return in_a.reshape((-1,) + in_a.shape[2:])

train_ims = val_reshape(train_ims)
train_locs = val_reshape(train_locs)
self.close_cursors()
#
def loss(pred_in, pred_out):
    return tf.nn.l2_loss(pred_in - pred_out)

self.cost = loss(self.pred, self.ph['y'])
self.create_optimizer()
self.init_and_restore(sess,True,['loss','dist'],iter)

sel_im = val_ims[sel[bb*bnum:bb*(bnum+1)],...]
sel_locs = val_locs[sel[bb*bnum:bb*(bnum+1)],...]
pred_locs = val_predlocs[sel[bb*bnum:bb*(bnum+1)],...]

in1 = PoseTools.scale_images(sel_im,2,conf)
self.fd[self.ph['x']] = in1
rescale = self.conf.unet_rescale
imsz = [self.conf.imsz[0] / rescale, self.conf.imsz[1] / rescale, ]
label_ims = PoseTools.create_label_images(
    sel_locs, imsz, 1, self.conf.label_blur_rad)
self.fd[self.ph['y']] = label_ims

c_pred = sess.run(self.pred,self.fd)

learning_rate=0.0001
ex_count = iter*conf.batch_size
cur_lr = learning_rate * self.conf.gamma ** math.floor(ex_count/ self.conf.step_size)
self.fd[self.ph['learning_rate']] = cur_lr
self.fd[self.ph['phase_train']] = True

for _ in range(100):
    sess.run(self.opt, self.fd)

self.fd[self.ph['phase_train']] = False

a_pred = sess.run(self.pred, self.fd)

#
val_file = os.path.join(self.conf.cachedir, self.conf.fulltrainfilename + '.tfrecords')
num_val = 0
for record in tf.python_io.tf_record_iterator(val_file):
    num_val += 1

# self.close_cursors()
self.create_cursors(sess)

val_dist_a = []
val_ims_a = []
val_preds_a = []
val_predlocs_a = []
val_locs_a = []
for step in range(num_val / self.conf.batch_size):
    # self.setup_val(sess)
    # self.read_images(self.DBType.Train, True, sess, False)
    # rescale = self.conf.unet_rescale
    # self.fd[self.ph['x']] = PoseTools.scale_images(
    #     self.xs, rescale, self.conf)
    # imsz = [self.conf.imsz[0] / rescale, self.conf.imsz[1] / rescale, ]
    # label_ims = PoseTools.create_label_images(
    #     self.locs / rescale, imsz, 1, self.conf.label_blur_rad)
    # self.fd[self.ph['y']] = label_ims

    cur_pred = sess.run(self.pred, self.fd)
    cur_predlocs = PoseTools.get_pred_locs(cur_pred)
    cur_dist = np.sqrt(np.sum(
        (cur_predlocs - self.locs/2) ** 2, 2))
    val_dist_a.append(cur_dist)
    val_ims_a.append(self.xs)
    val_locs_a.append(self.locs)
    val_preds_a.append(cur_pred)
    val_predlocs_a.append(cur_predlocs)


def val_reshape(in_a):
    in_a = np.array(in_a)
    return in_a.reshape((-1,) + in_a.shape[2:])

val_dist_a = val_reshape(val_dist_a)
val_ims_a = val_reshape(val_ims_a)
val_preds_a = val_reshape(val_preds_a)
val_predlocs_a = val_reshape(val_predlocs_a)
val_locs_a = val_reshape(val_locs_a)

mm = PoseTools.create_label_images(val_locs,[256,256],1,self.conf.label_blur_rad)
print(((mm-val_preds)**2).mean())
print(((mm-val_preds_a)**2).mean())
print(np.sum(val_dist_a>20,axis=0))
print(np.sum(val_dist>20,axis=0))

##
def loss(pred_in, pred_out):
    return tf.nn.l2_loss(pred_in - pred_out)

self.cost = loss(self.pred, self.ph['y'])

if not sess._closed:
    sess.close()
sess = tf.InteractiveSession()
self.init_and_restore(sess,True,['loss','dist'],iter)
ex_num = 3444
bnum = int(np.floor(ex_num/self.conf.batch_size))
ex_off = ex_num - bnum*self.conf.batch_size
# ll = self.down_layers + self.up_layers
ll = self.debug_layers
vv = tf.global_variables()
for ndx in reversed(range(len(vv))):
    if vv[ndx].name.find('weight')<0:
        vv.pop(ndx)
gg = tf.gradients(self.cost,vv)
for ndx in range(bnum+1):
    self.setup_val(sess)


ll_out = sess.run([ll, self.pred,gg],self.fd)
im1 = self.xs.copy()
for ndx in range(5):
    self.setup_val(sess)
ll_outt = sess.run([ll, self.pred,gg],self.fd)
pq = PoseTools.get_pred_locs(ll_outt[1])
dist_1 = np.sqrt(np.sum( (pq-self.locs/2)**2,axis=2))
im2 = self.xs.copy()

