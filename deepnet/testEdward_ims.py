from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import PoseTools

import edward as ed
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import math
import scipy
import copy

from edward.models import Categorical, Mixture, Normal, MultivariateNormalDiag
from tensorflow.contrib import slim
from scipy import stats
from sklearn.model_selection import train_test_split
import mymap

tf.reset_default_graph()
imsz = 128
# ed.set_seed(42)

N = 100
K = 5

#
def build_toy_dataset(N):
    k = 1
    sigma = 3
    all_locs = np.random.randint(4,imsz-3,[N,k,2])
    train_ims = (PoseTools.create_label_images(all_locs,
                                               [imsz,imsz], 1, sigma)+1)/2
    locs = np.zeros([N, 2])
    sample = np.zeros(N)
    mfac = 0.2 + np.random.random([N,k])
    mfac = mfac/mfac.sum(1)[:,np.newaxis]
    # lkl = 0
    for ndx in range(N):
        ri = np.random.choice(range(k),p=mfac[ndx,:])
        rands = np.random.randn(1,2)
        locs[ndx,:] = all_locs[ndx,ri,:] + rands*sigma/math.sqrt(2)
        sample[ndx] = ri
        # prob = scipy.stats.multivariate_normal.pdf(rands,[0,0],[[1,0],[0,1]]))
        # lkl += np.log(mfac[ndx,ri]*prob)
        # mfac[ndx,ri] = 0.6
    # train_ims = mfac[:,np.newaxis,np.newaxis,:] * train_ims
    train_ims = mfac[:,np.newaxis,np.newaxis,:] * \
                ((0.3+np.random.rand(N,1,1,1)) * train_ims)
    train_ims = train_ims.sum(-1)
    return train_ims, locs #, all_locs, sample



X_ph = tf.placeholder(tf.float32, [None,imsz,imsz])
y_ph = tf.placeholder(tf.float32, [None,2])
# y_y_ph = tf.placeholder(tf.float32, [None])

#
nnets = 8
def neural_network(X):
  """loc, scale, logits = NN(x; theta)"""
  # 2 hidden layers with 15 hidden units
  sz_x = X.get_shape().as_list()[2]
  sz_y = X.get_shape().as_list()[1]
  max_sz = max(sz_x,sz_y)
  X_flat = slim.flatten(X)
  hidden1 = slim.fully_connected(X_flat, 400,normalizer_fn=slim.batch_norm)
  hidden2 = slim.fully_connected(hidden1, 400,normalizer_fn=slim.batch_norm)
  locs = slim.fully_connected(hidden2, 2*K, activation_fn=None)
  # locs = tf.minimum(0., tf.maximum(locs,max_sz))
  o_scales = slim.fully_connected(hidden2, 2*K, activation_fn=tf.exp)
  scales = tf.minimum(3.,tf.maximum(2.,o_scales))
  logits = slim.fully_connected(hidden2, K, activation_fn=None)
  locs = tf.reshape(locs,[-1,K,2])
  scales = tf.reshape(scales,[-1,K,2])
  return locs, scales, logits, hidden1

def create_images(pred_means, pred_std, pred_weights, imsz, sel):
    out_test = np.zeros([imsz,imsz])
    for ndx in range(pred_means.shape[1]):
        cur_locs = pred_means[sel:sel+1,ndx:ndx+1,:].astype('int')
        cur_scale = pred_std[sel,ndx,:].mean().astype('int')
        curl = (PoseTools.create_label_images(cur_locs, [imsz, imsz], 1, cur_scale) + 1) / 2
        out_test += pred_weights[sel,ndx]*curl[0,...,0]
    return out_test


all_locs = []
all_scales = []
all_logits = []
psz = imsz/nnets
for xx in range(nnets):
    for yy in range(nnets):
        x_start = int(xx*psz)
        x_stop = int((xx+1)*psz)
        y_start = int(yy*psz)
        y_stop = int((yy+1)*psz)
        cur_patch = X_ph[:,y_start:y_stop,x_start:x_stop]
        with tf.variable_scope('nn') as scope:
            if xx or yy:
                scope.reuse_variables()
                locs, scales, logits,hidden1 = neural_network(cur_patch)
            else:
                locs, scales, logits,hidden1 = neural_network(cur_patch)
        locs += np.array([xx,yy])*(imsz/nnets)
        all_locs.append(locs)
        all_scales.append(scales)
        all_logits.append(logits)

all_locs = tf.concat(all_locs,1)
all_scales = tf.concat(all_scales,1)
all_logits = tf.concat(all_logits,1)

cat = Categorical(logits=all_logits)
components = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
              in zip(tf.unstack(tf.transpose(all_locs,[1,0,2])),
                     tf.unstack(tf.transpose(all_scales,[1,0,2])))]

y = Mixture(cat=cat, components=components,
            value=tf.zeros_like(y_ph))
# y_y = Mixture(cat=cat, components=components_y, value=tf.zeros_like(y_y_ph))
# Note: A bug exists in Mixture which prevents samples from it to have
# a shape of [None]. For now fix it using the value argument, as
# sampling is not necessary for MAP estimation anyways.

#

# There are no latent variables to infer. Thus inference is concerned
# with only training model parameters, which are baked into how we
# specify the neural networks.
n_epoch = 1000

#
inference = mymap.MAP(data={y: y_ph})
v_list = PoseTools.get_vars('nn')
inference.initialize(var_list=v_list,n_iter=n_epoch)

#

myOpt = True
starter_learning_rate = 0.001
step = tf.placeholder(tf.int64)
learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                           step, 300, 0.9, staircase=True)

if myOpt:
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(inference.loss)

sess = tf.InteractiveSession()
# sess = ed.get_session()
tf.global_variables_initializer().run()

train_loss = np.zeros(n_epoch)
train_loss_l2 = np.zeros(n_epoch)
test_loss = np.zeros(n_epoch)
test_loss_l2 = np.zeros(n_epoch)

X_all = []
l_all = []
for n in range(50):
    X_train, in_locs = build_toy_dataset(50)
    X_all.append(X_train)
    l_all.append(in_locs)
#
count = 0
from stephenHeadConfig import conf
conf.trange = 10
conf.rrange = 90
for i in range(n_epoch):
    count = (count+1) % len(X_all)
    X_test, test_locs = build_toy_dataset(50)
    X_train = copy.deepcopy(X_all[count][:,np.newaxis,...])
    in_locs = copy.deepcopy(l_all[count][:,np.newaxis,...])
    X_train, in_locs = PoseTools.randomly_translate(X_train, in_locs, conf)
    X_train, in_locs = PoseTools.randomly_rotate(X_train, in_locs, conf)
    X_train = X_train[:,0,...]
    in_locs = in_locs[:,0,...]

    if myOpt:
        sess.run(optimizer,feed_dict={X_ph:X_train,
                                      y_ph:in_locs,
                                      step:i})
        train_loss[i] = sess.run(inference.loss,
                              feed_dict={X_ph: X_train,
                                        y_ph: in_locs})
        test_loss[i] = sess.run(inference.loss,
                              feed_dict={X_ph: X_test,
                                        y_ph: test_locs})
        if i%50==0:
            pred_weights, pred_means, pred_std = \
                sess.run([tf.nn.softmax(all_logits),
                          all_locs, all_scales],
                         feed_dict={X_ph: X_test})
            test_dist = 0
            for count in range(10):
                sel = np.random.randint(50)
                out_test = create_images(pred_means,
                                         pred_std,
                                         pred_weights,
                                         imsz, sel)
                maxndx = np.argmax(out_test)
                predloc = np.array(np.unravel_index(
                    maxndx, out_test.shape))
                predloc_1 = predloc.copy()
                predloc_1[0] = predloc[1]
                predloc_1[1] = predloc[0]
                tt1 = np.sqrt(np.sum((test_locs[sel, :] - predloc_1) ** 2))
                test_dist += tt1
            print('{}: {:.4f}, {:.2f}'.format(i, test_loss[i], test_dist/10))
    else:
        info_dict = inference.update(feed_dict={X_ph: X_train,
                                                y_ph: in_locs})
        train_loss[i] = info_dict['loss']
        test_loss[i] = sess.run(inference.loss,
                              feed_dict={X_ph: X_test,
                                        y_ph: test_locs})
        inference.print_progress(info_dict)

    if i%20 == 0:
        pred_weights, pred_means, pred_std = \
            sess.run([tf.nn.softmax(all_logits), all_locs, all_scales], feed_dict={X_ph: X_test})
        l2_dist = 0
        for sel in range(X_test.shape[0]):
            out_test = np.zeros([imsz, imsz])
            for ndx in range(pred_means.shape[1]):
                cur_locs = pred_means[sel:sel + 1, ndx:ndx + 1, :].astype('int')
                cur_scale = pred_std[sel, ndx, :].mean().astype('int')
                curl = (PoseTools.create_label_images(cur_locs, [imsz, imsz], 1, cur_scale) + 1) / 2
                out_test += pred_weights[sel, ndx] * curl[0, ..., 0]
            l2_dist += ((X_test[sel, ...] - out_test) ** 2).sum()

        test_loss_l2[i] = math.sqrt(l2_dist / X_test.size)


#
X_test, in_locs = build_toy_dataset(500)

pred_weights, pred_means, pred_std = \
    sess.run([tf.nn.softmax(all_logits), all_locs, all_scales],
             feed_dict={X_ph: X_test})

l2_dist = 0
for sel in range(X_test.shape[0]):
    out_test = np.zeros([imsz,imsz])
    for ndx in range(pred_means.shape[1]):
        cur_locs = pred_means[sel:sel+1,ndx:ndx+1,:].astype('int')
        cur_scale = pred_std[sel,ndx,:].mean().astype('int')
        curl = (PoseTools.create_label_images(cur_locs, [imsz, imsz], 1, cur_scale) + 1) / 2
        out_test += pred_weights[sel,ndx]*curl[0,...,0]
    l2_dist += ((X_test[sel,...]-out_test)**2).sum()

l2_dist = math.sqrt(l2_dist/X_test.size)
plt.plot(test_loss_l2)
##


#

sel = np.random.randint(50)
fig = plt.figure(1)
fig.clear()
ax = fig.add_subplot(1,2,1)
ax.imshow(X_test[sel,...],interpolation='nearest')
for ndx in range(pred_means.shape[1]):
    ell = matplotlib.patches.Ellipse(xy=pred_means[sel,ndx,:],
                                     width=pred_std[sel,ndx,0]*3,
                                     height=pred_std[sel,ndx,1]*3,
                                     angle=0,lw=2.)
    ax.add_artist(ell)
    ell.set_alpha(pred_weights[sel,ndx]/2)

out_test = create_images(pred_means,pred_std,pred_weights,
                         imsz,sel)
ax = fig.add_subplot(1,2,2)
ax.imshow(out_test,interpolation='nearest')

    # fig.savefig('/home/mayank/temp/gmm_{}.png'.format(count+1))


## profile the above code

X_test, test_locs = build_toy_dataset(500)
from tensorflow.tools.tfprof import tfprof_log_pb2
run_metadata = tf.RunMetadata()

loss_value = sess.run(inference.loss, feed_dict = {X_ph: X_test, y_ph: test_locs},
        options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        run_metadata=run_metadata)

op_log = tfprof_log_pb2.OpLog()

opts = tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY.copy()
# opts['output'] = 'timeline:outfile=/home/mayank/temp/test_tfprof'

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
