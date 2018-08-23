##

import tensorflow as tf
import PoseMDN
from stephenHeadConfig import conf as conf
import PoseTools
import edward as ed
import math
import numpy as np
import matplotlib.pyplot as plt
import mymap

tf.reset_default_graph()
restore = True
trainType = 0
conf.batch_size = 16
conf.nfcfilt = 256
self = PoseMDN.PoseMDN(conf)
self.conf.expname += '_normal_train'
##
# self.conf.expname += '_normal_with_base'
self.train_offline(restore=True,trainType=trainType)
##

self.conf.trange = self.conf.imsz[0] // 25

mdn_dropout = 1.
self.create_ph()
self.createFeedDict()
self.feed_dict[self.ph['keep_prob']] = mdn_dropout
self.feed_dict[self.ph['phase_train_base']] = False
self.feed_dict[self.ph['phase_train_mdn']] = False
self.trainType = trainType

y = self.create_network()
self.openDBs()
self.createBaseSaver()
self.create_saver()

y_label = self.ph['base_locs'] / self.conf.rescale / self.conf.pool_scale
data_dict = {}
for ndx in range(self.conf.n_classes):
    data_dict[y[ndx]] = y_label[:, ndx, :]
inference = mymap.MAP(data=data_dict)
inference.initialize(var_list=PoseTools.get_vars('mdn'))
self.loss = inference.loss

starter_learning_rate = 0.00003
decay_steps = 5000 / 8 * self.conf.batch_size
learning_rate = tf.train.exponential_decay(
    starter_learning_rate, self.ph['step'], decay_steps, 0.9,
    staircase=True)

self.opt = tf.train.AdamOptimizer(
    learning_rate=learning_rate).minimize(self.loss)

sess = tf.InteractiveSession()
self.createCursors(sess)
self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
sess.run(tf.global_variables_initializer())
self.restoreBase(sess, restore=True)
self.restore(sess, restore)

##

self.updateFeedDict(self.DBType.Val, sess=sess,
                    distort=False)
self.feed_dict[self.ph['keep_prob']] = mdn_dropout
self.feed_dict[self.ph['learning_rate']] = 0.001
self.feed_dict[self.ph['step']] = 1

#
pred_weights, pred_means, pred_std = \
    sess.run([tf.nn.softmax(self.mdn_logits,dim=1),
              self.mdn_locs, self.mdn_scales],
             feed_dict=self.feed_dict)


osz = conf.imsz[0] // 4
l2_dist = 0
sel = np.random.randint(conf.batch_size)
out_test = np.zeros([osz,osz,conf.n_classes])
f = plt.figure()
ax = f.add_subplot(2,3,1)
ax.imshow(self.xs[sel,0,...], cmap='gray')
ax.scatter(self.locs[sel,:,0], self.locs[sel,:,1], marker='.')
for cls in range(conf.n_classes):
    for ndx in range(pred_means.shape[1]):
        cur_locs = pred_means[sel:sel+1,ndx:ndx+1,cls,:].astype('int')
        cur_scale = pred_std[sel,ndx,cls,:].mean().astype('int')
        curl = (PoseTools.create_label_images(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
        out_test[:,:,cls] += pred_weights[sel,ndx,cls]*curl[0,...,0]

    ax = f.add_subplot(2,3,cls+2)
    ax.imshow(out_test[...,cls], interpolation='nearest')#vmin=0., vmax=1.)
    ax.scatter(self.locs[sel,cls,0]/4, self.locs[sel,cls,1]/4, marker='.')


##

def create_mdn_pred(self, pred_weights, pred_means, pred_std):
    conf = self.conf
    osz = self.conf.imsz[0] // self.conf.rescale // self.conf.pool_scale
    mdn_pred_out = np.zeros([self.conf.batch_size, osz, osz, conf.n_classes])
    for sel in range(conf.batch_size):
        for cls in range(conf.n_classes):
            for ndx in range(pred_means.shape[1]):
                if pred_weights[sel,ndx,cls] < 0.02:
                    continue
                cur_locs = pred_means[sel:sel + 1, ndx:ndx + 1, cls, :].astype('int')
                cur_scale = pred_std[sel, ndx, cls, :].mean().astype('int')
                curl = (PoseTools.create_label_images(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
                mdn_pred_out[sel,:, :, cls] += pred_weights[sel, ndx, cls] * curl[0, ..., 0]
    return  mdn_pred_out

val_dist = 0.

allinfo = []
for count in range(20):
    self.updateFeedDict(self.DBType.Val, sess=sess,
                        distort=False)
    self.updateFeedDict(self.DBType.Val, sess=sess,
                        distort=False)
    self.updateFeedDict(self.DBType.Val, sess=sess,
                        distort=False)
    self.updateFeedDict(self.DBType.Val, sess=sess,
                        distort=False)
    self.feed_dict[self.ph['keep_prob']] = 1.
    self.feed_dict[self.ph['learning_rate']] = 0.001
    self.feed_dict[self.ph['step']] = 1

    pred_weights, pred_means, pred_std = \
        sess.run([tf.nn.softmax(self.mdn_logits,dim=1),
                  self.mdn_locs, self.mdn_scales],
                 feed_dict=self.feed_dict)
    bpred = sess.run(self.basePred, self.feed_dict)
    lbl_imgs = self.feed_dict[self.ph['y']]

    mdn_pred = create_mdn_pred(self,pred_weights,pred_means,pred_std)
    bee = PoseTools.get_base_error(self.locs, mdn_pred, self.conf)
    tt1 = np.sqrt(np.sum(np.square(bee), 2))
    allinfo.append([tt1,self.xs,self.locs,
                    pred_weights,pred_means,
                    pred_std,mdn_pred,
                    bpred,lbl_imgs])
    nantt1 = np.invert(np.isnan(tt1.flatten()))
    val_dist += tt1.flatten()[nantt1].mean()

print(val_dist/(count+1))

##

alltinfo = []
val_dist = 0.
for count in range(10):
    self.updateFeedDict(self.DBType.Train, sess=sess,
                        distort=True)
    self.updateFeedDict(self.DBType.Train, sess=sess,
                        distort=True)
    self.updateFeedDict(self.DBType.Train, sess=sess,
                        distort=True)
    self.updateFeedDict(self.DBType.Train, sess=sess,
                        distort=True)
    self.feed_dict[self.ph['keep_prob']] = 0.95
    self.feed_dict[self.ph['learning_rate']] = 0.001
    self.feed_dict[self.ph['step']] = 1

    pred_weights, pred_means, pred_std = \
        sess.run([tf.nn.softmax(self.mdn_logits,dim=1),
                  self.mdn_locs, self.mdn_scales],
                 feed_dict=self.feed_dict)
    bpred = sess.run(self.basePred, self.feed_dict)
    lbl_imgs = self.feed_dict[self.ph['y']]

    mdn_pred = create_mdn_pred(self,pred_weights,pred_means,pred_std)
    bee = PoseTools.get_base_error(self.locs, mdn_pred, self.conf)
    tt1 = np.sqrt(np.sum(np.square(bee), 2))
    alltinfo.append([tt1,self.xs,self.locs,
                    pred_weights,pred_means,
                    pred_std,mdn_pred,
                    bpred,lbl_imgs])
    nantt1 = np.invert(np.isnan(tt1.flatten()))
    val_dist += tt1.flatten()[nantt1].mean()

print(val_dist/(count+1))
bout = np.array([a[-2] for a in alltinfo]).reshape([-1,128,128,5])
gg = bout.max(axis=(1,2))
plt.figure(); plt.hist(gg,normed=True)

##


##
from tensorflow.tools.tfprof import tfprof_log_pb2
run_metadata = tf.RunMetadata()
self.updateFeedDict(self.DBType.Val, sess=sess,
                    distort=True)
self.feed_dict[self.ph['keep_prob']] = mdn_dropout
self.feed_dict[self.ph['learning_rate']] = 0.001
self.feed_dict[self.ph['step']] = 1


loss_value = sess.run(self.opt, feed_dict = self.feed_dict,
        options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        run_metadata=run_metadata)

op_log = tfprof_log_pb2.OpLog()

opts = tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY.copy()
opts['output'] = 'timeline:outfile=/home/mayank/temp/mdn_tfprof'

tf.contrib.tfprof.tfprof_logger.write_op_log(
        tf.get_default_graph(),
        log_dir="/tmp/log_dir",
        op_log=op_log,
        run_meta=run_metadata)

bb = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        run_meta=run_metadata,
        op_log=op_log,
        tfprof_cmd='code',
        tfprof_options=opts)

##

all_sel = 57
sel1 = all_sel//conf.batch_size
sel = all_sel%conf.batch_size

curxs = allinfo[sel1][1]
curlocs = allinfo[sel1][2]
pred_weights = allinfo[sel1][3]
pred_means = allinfo[sel1][4]
pred_std = allinfo[sel1][5]
mdn_pred = allinfo[sel1][6]
bpred = allinfo[sel1][7]

f = plt.figure()
ax = f.add_subplot(2,6,1)
ax.imshow(curxs[sel,0,...], cmap='gray')
ax.scatter(curlocs[sel,:,0], curlocs[sel,:,1], marker='.')
for cls in range(conf.n_classes):

    ax = f.add_subplot(2,6,cls+2)
    ax.imshow(mdn_pred[sel,...,cls], interpolation='nearest',
              vmin=-1., vmax=1., cmap='jet')
    ax.grid(False)
    ax.scatter(curlocs[sel,cls,0]/4, curlocs[sel,cls,1]/4,
               marker='.', s=40)

    ax = f.add_subplot(2,6,6+cls+2)
    ax.imshow(bpred[sel,...,cls], interpolation='nearest',
              vmin=-1., vmax=1.,cmap='jet')
    ax.grid(False)
    # ax.scatter(curlocs[sel,cls,0]/4, curlocs[sel,cls,1]/4,
    #            marker='.', s=40, c='g')

##

alltt = np.array([a[0] for a in alltinfo]).reshape([-1,5])
vv = np.flipud(np.argsort(alltt.sum(axis=1)))
all_sel = vv[np.random.randint(7)]
sel1 = all_sel//conf.batch_size
sel = all_sel%conf.batch_size

curxs = alltinfo[sel1][1]
curlocs = alltinfo[sel1][2]
bpred = alltinfo[sel1][7]
lbl = alltinfo[sel1][8]

f = plt.figure()
ax = f.add_subplot(2,6,1)
ax.imshow(curxs[sel,0,...], cmap='gray')
ax.scatter(curlocs[sel,:,0], curlocs[sel,:,1], marker='.')
for cls in range(conf.n_classes):

    ax = f.add_subplot(2,6,cls+2)
    ax.imshow(mdn_pred[sel,...,cls], interpolation='nearest',
              vmin=-1., vmax=1.,cmap='jet')
    ax.grid(False)
    ax.scatter(curlocs[sel,cls,0]/4, curlocs[sel,cls,1]/4,
               marker='.', s=40)

    ax = f.add_subplot(2,6,6+cls+2)
    ax.imshow(lbl[sel,...,cls], interpolation='nearest',
              vmin=-1., vmax=1.,cmap='jet')
    ax.grid(False)
    # ax.scatter(curlocs[sel,cls,0]/4, curlocs[sel,cls,1]/4,
    #            marker='.', s=40, c='g')


##

curinfo = allinfo
alocs = np.array([a[2] for a in curinfo]).reshape([-1,5,2])
ahmap = np.array([a[7] for a in curinfo]).reshape([-1,128,128,5])

mean_maps = np.zeros([40,40,5])
for ndx1 in range(alocs.shape[0]):
    for ndx2 in range(alocs.shape[1]):
        xloc = int(alocs[ndx1,ndx2,0]/4)
        yloc = int(alocs[ndx1,ndx2,1]/4)
        curpatch = ahmap[ndx1,yloc-20:yloc+20,xloc-20:xloc+20,ndx2]
        mean_maps[:,:,ndx2] += curpatch

