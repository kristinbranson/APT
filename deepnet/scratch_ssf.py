import tensorflow as tf
import PoseMDN
from stephenHeadConfig import conf as conf
import PoseTools
import math
import numpy as np
import matplotlib.pyplot as plt
import PoseTrain
import os
import pickle
import copy
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

conf.mdn_min_sigma = 3.
conf.mdn_max_sigma = 4.
conf.save_step = 50
conf.maxckpt = 100
net = 'nvgg'

if net is 'vgg':
    conf.expname = 'head_joint_vgg'
else:
    conf.expname = 'head_joint_5layers_ssf_lrx50'

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

if net is 'vgg':
    self.create_network_vgg(self.ph['base_pred'])
else:
    self.create_network_joint(self.ph['base_pred'])

self.loss = self.my_loss_joint()

self.openDBs()
self.create_saver()

starter_learning_rate = 0.0001
decay_steps = 5000 * 8 / self.conf.batch_size
learning_rate = tf.train.exponential_decay(
    starter_learning_rate, self.ph['step'], decay_steps, 0.9,
    staircase=True)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    self.opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(self.loss)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)

self.createCursors(sess)
self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
self.feed_dict[self.ph['base_locs']] = np.zeros([self.conf.batch_size,
                                                 self.conf.n_classes, 2])
sess.run(tf.global_variables_initializer())

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
m_test_data = o_test_data

self.restore(sess, restore)

def PCA(data, dims_rescaled_data=2):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    import numpy as NP
    from scipy import linalg as LA
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = NP.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = LA.eigh(R)
    # sort eigenvalue in decreasing order
    idx = NP.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return NP.dot(evecs.T, data.T).T, evals, evecs

##

vnum = 20
ckpt_filename = self.conf.expname + '_MDN_ckpt'
latest_ckpt = tf.train.get_checkpoint_state(self.conf.cachedir,
                                            latest_filename=ckpt_filename)

kk = latest_ckpt.all_model_checkpoint_paths
kk = kk[::-1]
vv = tf.global_variables()

self.mdn_saver.restore(sess,kk[0])
gg1 = vv[vnum].eval(sess)
self.mdn_saver.restore(sess,kk[30])
gg2 = vv[vnum].eval(sess)