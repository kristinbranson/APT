# # ##
# #
# import tensorflow as tf
# import PoseMDN
# from stephenHeadConfig import conf as conf
# import PoseTools
# import edward as ed
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import mymap
# import PoseTrain
# import os
# import pickle
# from tensorflow.contrib import slim
# from edward.models import Categorical, Mixture, Normal, MultivariateNormalDiag
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# conf.expname = 'head_tf1p3'
# conf.baseckptname = 'head_tf1p3_Baseckpt'
# conf.baseoutname =  'head_tf1p3_Base'
# conf.basedataname =  'head_tf1p3_Basetraindata'
#
# tf.reset_default_graph()
#
# restore = True
# trainType = 0
# conf.batch_size = 16
# # conf.nfcfilt = 256
# self = PoseTrain.PoseTrain(conf)
# self.createPH()
# self.createFeedDict()
# self.feed_dict[self.ph['keep_prob']] = 1.
# self.feed_dict[self.ph['phase_train_base']] = False
# self.feed_dict[self.ph['learning_rate']] = 0.
# self.trainType = 0
#
# with tf.variable_scope('base'):
#     self.createBaseNetwork(doBatch=True)
#
# self.conf.trange = self.conf.imsz[0] // 25
#
# self.openDBs()
# self.createBaseSaver()
#
# sess = tf.InteractiveSession()
# self.createCursors(sess)
# self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
# sess.run(tf.global_variables_initializer())
# self.restoreBase(sess, restore=True)
#
# #
#
# train_file = os.path.join(conf.cachedir,
#                          conf.trainfilename) + '.tfrecords'
# num_train = 0
# for t in tf.python_io.tf_record_iterator(train_file):
#     num_train += 1
#
# test_file = os.path.join(conf.cachedir,
#                          conf.valfilename) + '.tfrecords'
# num_test = 0
# for t in tf.python_io.tf_record_iterator(test_file):
#     num_test += 1
#
# train_data = []
# for ndx in range(num_train//conf.batch_size):
#     self.updateFeedDict(self.DBType.Train,distort=True,
#                         sess=sess)
#     cur_pred = sess.run(self.basePred, self.feed_dict)
#     train_data.append([cur_pred, self.locs, self.info, self.xs])
#
# test_data = []
# for ndx in range(num_test//conf.batch_size):
#     self.updateFeedDict(self.DBType.Val,distort=False,
#                         sess=sess)
#     cur_pred = sess.run(self.basePred, self.feed_dict)
#     test_data.append([cur_pred, self.locs, self.info, self.xs])
#
# with open(os.path.join(self.conf.cachedir, 'base_predictions'),
#           'wb') as f:
#     pickle.dump([train_data,test_data], f, protocol=2)
#

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
import PoseTrain
import os
import pickle
from tensorflow.contrib import slim
from edward.models import Categorical, Mixture, Normal, MultivariateNormalDiag
import copy
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ntype = 'conv'

conf.mdn_min_sigma = 3.
conf.mdn_max_sigma = 4.
if ntype is 'conv':
    conf.expname = 'head_gridconv'
else:
    conf.expname = 'head_myloss_unboundlocs'


kk=0
# conf.expname = 'head_train_jitter_sigma3to4_my_bnorm_blocs'

tf.reset_default_graph()
restore = True
trainType = 0
conf.batch_size = 16
conf.nfcfilt = 256
self = PoseMDN.PoseMDN(conf)

mdn_dropout = 0.95
self.create_ph()
self.createFeedDict()
self.feed_dict[self.ph['keep_prob']] = mdn_dropout
self.feed_dict[self.ph['phase_train_base']] = False
self.feed_dict[self.ph['phase_train_mdn']] = True
self.feed_dict[self.ph['learning_rate']] = 0.
self.trainType = trainType

self.ph['base_pred'] = tf.placeholder(
    tf.float32, [None, 128, 128, self.conf.n_classes])

y_label = self.ph['base_locs'] / self.conf.rescale / self.conf.pool_scale
self.mdn_label = y_label

if ntype is 'conv':
    self.create_network(self.ph['base_pred'])
else:
    self.create_network_grid(self.ph['base_pred'])

self.loss = self.my_loss()

# y = self.create_network_ed(self.ph['base_pred'])
# self.mdn_out = y
#
# data_dict = {}
# for ndx in range(self.conf.n_classes):
#     data_dict[y[ndx]] = y_label[:, ndx, :]
# inference = mymap.MAP(data=data_dict)
# inference.initialize(var_list=PoseTools.getvars('mdn'))
# self.loss = inference.loss

self.openDBs()
self.create_saver()

# # old settings
starter_learning_rate = 0.0001
decay_steps = 5000 * 8 / self.conf.batch_size
learning_rate = tf.train.exponential_decay(
    starter_learning_rate, self.ph['step'], decay_steps, 0.9,
    staircase=True)

# # new settings
# starter_learning_rate = 0.0001
# decay_steps = 12000 * 8 / self.conf.batch_size
# learning_rate = tf.train.exponential_decay(
#     starter_learning_rate, self.ph['step'], decay_steps, 0.9)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    self.opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(self.loss)

# adam_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
# self.opt =slim.learning.create_train_op(
#     self.loss, adam_opt, global_step=step)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)

self.createCursors(sess)
self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
self.feed_dict[self.ph['base_locs']] = np.zeros([self.conf.batch_size,
                                                 self.conf.n_classes, 2])
sess.run(tf.global_variables_initializer())
# self.initializeRemainingVars(sess)

with open(os.path.join(conf.cachedir, 'base_predictions'),
          'rb') as f:
    train_data, test_data = pickle.load(f)

m_train_data = train_data
o_test_data = copy.deepcopy(test_data)
for ndx in range(len(test_data)):
    bpred_in = test_data[ndx][0]
    tr_ndx = np.random.choice(len(train_data))
    bpred_tr = train_data[tr_ndx][0]
    bpred_out = -1 * np.ones(bpred_in.shape)
    tr_locs = PoseTools.get_base_pred_locs(bpred_tr, self.conf) / 4
    in_locs = PoseTools.get_base_pred_locs(bpred_in, self.conf) / 4

    # training blobs test locations
    # for ex in range(bpred_in.shape[0]):
    #     for cls in range(bpred_in.shape[-1]):
    #         cur_sc = bpred_in[ex,:,:,cls]
    #         inxx = int(in_locs[ex,cls,0])
    #         inyy = int(in_locs[ex,cls,1])
    #         trxx = int(tr_locs[ex,cls,0])
    #         tryy = int(tr_locs[ex,cls,1])
    #         tr_sc = bpred_tr[ex,(tryy-10):(tryy+10),
    #                 (trxx - 10):(trxx + 10),cls]
    #         bpred_out[ex,(inyy-10):(inyy+10),
    #             (inxx - 10):(inxx + 10),cls ] = tr_sc

    # test blobs training locs
    for ex in range(bpred_in.shape[0]):
        for cls in range(bpred_in.shape[-1]):
            cur_sc = bpred_in[ex,:,:,cls]
            inxx = int(in_locs[ex,cls,0])
            inyy = int(in_locs[ex,cls,1])
            trxx = int(tr_locs[ex,cls,0])
            tryy = int(tr_locs[ex,cls,1])
            test_sc = bpred_in[ex,(inyy-10):(inyy+10),
                    (inxx - 10):(inxx + 10),cls]
            bpred_out[ex,(tryy-10):(tryy+10),
                (trxx - 10):(trxx + 10),cls ] = test_sc
    test_data[ndx][0] = bpred_out
# m_test_data = [i for sl in zip(test_data,o_test_data) for i in sl]
m_test_data = o_test_data

self.restore(sess, restore)

##

self.updateFeedDict(self.DBType.Train, sess=sess,
                    distort=True)

mdn_steps = 50000 * 8 / self.conf.batch_size
test_step = 0
self.feed_dict[self.ph['phase_train_mdn']] = True
train_time = 0
test_time = 0
for cur_step in range(self.mdn_start_at, mdn_steps):
    start = time.time()
    self.feed_dict[self.ph['step']] = cur_step
    data_ndx = cur_step%len(m_train_data)
    cur_bpred = m_train_data[data_ndx][0]
    pd = 15
    cur_bpred = np.pad(cur_bpred,[[0,0],[pd,pd],[pd,pd],[0,0]],
                       'constant',constant_values=-1)
    dxx = np.random.randint(pd*2)
    dyy = np.random.randint(pd*2)
    cur_bpred = cur_bpred[:,dyy:(128+dyy),dxx:(128+dxx),:]
    # self.feed_dict[self.ph['step']] = cur_step
    self.feed_dict[self.ph['base_locs']] = \
        PoseTools.get_base_pred_locs(cur_bpred, self.conf)
    self.feed_dict[self.ph['base_pred']] = cur_bpred
    sess.run(self.opt, self.feed_dict)
    train_time = 0.05 * (time.time()-start) + 0.95 * train_time

    if cur_step % self.conf.display_step == 0:
        start = time.time()
        data_ndx = (cur_step+1) % len(m_train_data)
        cur_bpred = m_train_data[data_ndx][0]
        self.feed_dict[self.ph['base_locs']] = \
            PoseTools.get_base_pred_locs(cur_bpred, self.conf)
        self.feed_dict[self.ph['base_pred']] = cur_bpred
        tr_loss = sess.run(self.loss, feed_dict=self.feed_dict)
        self.mdn_train_data['train_err'].append(tr_loss)
        self.mdn_train_data['step_no'].append(cur_step)

        mdn_pred = self.mdn_pred(sess)
        bee = PoseTools.get_base_error(
            self.feed_dict[self.ph['base_locs']], mdn_pred, conf)
        tt1 = np.sqrt(np.sum(np.square(bee), 2))
        nantt1 = np.invert(np.isnan(tt1.flatten()))
        train_dist = tt1.flatten()[nantt1].mean()
        self.mdn_train_data['train_dist'].append(train_dist)

        data_ndx = (test_step + 1) % len(m_test_data)
        cur_bpred = m_test_data[data_ndx][0]
        self.feed_dict[self.ph['base_locs']] = \
            PoseTools.get_base_pred_locs(cur_bpred, self.conf)
        self.feed_dict[self.ph['base_pred']] = cur_bpred
        cur_te_loss = sess.run(self.loss, feed_dict=self.feed_dict)
        val_loss = cur_te_loss
        mdn_pred = self.mdn_pred(sess)
        bee = PoseTools.get_base_error(
            self.feed_dict[self.ph['base_locs']],mdn_pred,conf)
        tt1 = np.sqrt(np.sum(np.square(bee), 2))
        nantt1 = np.invert(np.isnan(tt1.flatten()))
        val_dist = tt1.flatten()[nantt1].mean()
        test_step += 1

        self.mdn_train_data['val_err'].append(val_loss)
        self.mdn_train_data['val_dist'].append(val_dist)
        test_time = 0.05 * (time.time()-start) + 0.95 * test_time

        print('{}:Train Loss:{:.4f},{:.2f}, Test Loss:{:.4f},{:.2f}, Time:{:.4f},{:.4f}'.format(
            cur_step, tr_loss, train_dist, val_loss, val_dist,train_time,test_time))

    if cur_step % self.conf.save_step == 0:
        self.save(sess, cur_step)

print("Optimization finished!")
self.save(sess, mdn_steps)
self.closeCursors()


##
