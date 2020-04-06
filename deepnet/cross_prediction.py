from poseConfig import aliceConfig as conf
import os
import PoseTools
import multiResData
import tensorflow as tf
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
train_db =  os.path.join(conf.cachedir,conf.trainfilename) + '.tfrecords'
val_db =  os.path.join(conf.cachedir,conf.valfilename) + '.tfrecords'

tims, tlocs, tinfo = multiResData.read_and_decode_without_session(train_db,conf,())
vims, vlocs, vinfo = multiResData.read_and_decode_without_session(val_db,conf,())

tims = np.array(tims)
tlocs = np.array(tlocs)
tinfo = np.array(tinfo)
vims = np.array(vims)
vlocs = np.array(vlocs)
vinfo = np.array(vinfo)
vlocs[1106,...] = vlocs[:1000,:,:].mean(axis=0)
#

pt_2_remove = 13
locs_ph = tf.placeholder(tf.float32,[None,conf.n_classes-1,2])
pred_ph = tf.placeholder(tf.float32,[None,2])
step = tf.placeholder(tf.int16)

l_f = tf.contrib.layers.flatten(locs_ph)
l1 = tf.contrib.layers.fully_connected(l_f,24)
l2 = tf.contrib.layers.fully_connected(l1,18)
l3 = tf.contrib.layers.fully_connected(l2,12)
l4 = tf.contrib.layers.fully_connected(l3,8)
l_out = tf.contrib.layers.fully_connected(l4,2,activation_fn=None)

loss = tf.nn.l2_loss(pred_ph-l_out)

rate = tf.train.exponential_decay(0.01, step, 5000, 0.1)
train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss=loss)

bsize = 16
n_train = tims.shape[0]
n_val = vims.shape[0]

#
sess = tf.Session()

init_op = tf.initialize_all_variables()
sess.run(init_op)
fd = {}
t_ndx = range(pt_2_remove) + range(pt_2_remove+1,conf.n_classes)

#
all_loss = []
all_dist = []
for cur_step in range(20000):
    fd[step] = cur_step
    cur_ndx = cur_step % (n_train/bsize)
    cur_locs = tlocs[cur_ndx*bsize: (cur_ndx+1)*bsize,...]
    cur_ims = tims[cur_ndx*bsize: (cur_ndx+1)*bsize,...]
    # mod_ims, mod_locs = PoseTools.preprocess_ims(cur_ims,cur_locs,conf,True,1)
    mod_ims = cur_ims
    mod_locs = cur_locs + np.random.randn(2)*10
    zz = mod_locs[:,t_ndx,:] - np.array(conf.imsz)/2
    fd[locs_ph] = zz
    fd[pred_ph] = mod_locs[:,pt_2_remove,:] - np.array(conf.imsz)/2

    sess.run(train_op, fd)
    if cur_step % 100 ==0:
        cur_loss, cur_pred = sess.run([loss,l_out],fd)
        dist = np.mean(np.sqrt(np.sum( (fd[pred_ph] - cur_pred)**2 ,axis=-1)))
        print('Step:{}, Loss:{}, Dist:{}'.format(cur_step,cur_loss, dist))
        all_loss.append(cur_loss)
        all_dist.append(dist)

#

preds = np.zeros([n_val,2])
for v_ndx in range(n_val/bsize):
    cur_locs = vlocs[v_ndx*bsize: (v_ndx+1)*bsize,...]
    cur_ims = tims[v_ndx*bsize: (v_ndx+1)*bsize,...]
#    mod_ims, mod_locs = PoseTools.preprocess_ims(cur_ims,cur_locs,conf,True,1)
    zz = cur_locs[:,t_ndx,:]
    fd[locs_ph] = zz  - np.array(conf.imsz)/2
    fd[pred_ph] = cur_locs[:,pt_2_remove,:] - np.array(conf.imsz)/2

    cur_pred = sess.run(l_out, fd)
    preds[v_ndx*bsize: (v_ndx+1)*bsize,:] = cur_pred + np.array(conf.imsz)/2

dd = np.sqrt(np.sum( (preds-vlocs[:,pt_2_remove,:])**2 ,axis=-1))
dd = dd[:1152]
print(dd[:1152].mean())

##
import pickle

with open('/home/mayank/temp/alice_val_results_mdn.p', 'r') as f:
    ii,pp = pickle.load(f)

ff = np.where( (ii[:,pt_2_remove]-dd )> 4)[0]

##
curndx = np.random.choice(ff)
plt.imshow(vims[curndx,:,:,0],'gray')
plt.scatter(preds[curndx,0],preds[curndx,1],c='r')
plt.scatter(vlocs[curndx,t_ndx,0],vlocs[curndx,t_ndx,1],c='g',marker='+')
plt.scatter(vlocs[curndx,pt_2_remove,0],vlocs[curndx,pt_2_remove,1],c='g')
plt.scatter(pp[curndx,pt_2_remove,0],pp[curndx,pt_2_remove,1],c='b')