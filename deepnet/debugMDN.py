


device = '0'
import os
os.environ['CUDA_VISIBLE_DEVICES'] = device

from stephenHeadConfig import conf
conf.cachedir = '/home/mayank/temp/cacheHead_Round2'

import PoseUMDN
import tensorflow as tf
import PoseTools
import math
import PoseCommon


self = PoseUMDN.PoseUMDN(conf,unet_name='pose_unet_nomean_contrast')
# self.train_umdn(False,0)
self.init_train(0)
self.pred = self.create_network()
self.cost = self.my_loss(self.pred, self.ph['locs'])
self.create_optimizer()
self.create_saver()

sess = tf.InteractiveSession()
start_at = self.init_and_restore(
    sess, False, ['loss','dist'])
self.fd[self.ph['learning_rate']] = 0.0001
self.fd_train()
self.update_fd(self.DBType.Train, sess, True)

##

import pickle
import os
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt


# ind_loss = []
# for ndx in range(conf.n_classes):
#     ind_loss.append(inference.ind_loss[y[ndx]])
# ind_loss = tf.transpose(tf.stack(ind_loss), [1, 0])

ind_loss = []
for cls in range(self.conf.n_classes):
    ll = tf.nn.softmax(self.mdn_logits[..., cls], dim=1)

    cur_scales = self.mdn_scales[:, :, cls]
    pp = self.mdn_label[:, cls:cls + 1, :]
    kk = tf.reduce_sum(tf.square(pp - self.mdn_locs[:, :, cls, :]), axis=2)
    cur_comp = tf.div(tf.exp(-kk / (cur_scales ** 2) / 2), 2 * np.pi * (cur_scales ** 2))

    # cur_comp = [i.prob(self.mdn_label[:,cls,:]) for i in self.components[cls]]
    # cur_comp = tf.stack(cur_comp)
    pp = cur_comp * ll
    cur_loss = tf.log(tf.reduce_sum(pp, axis=1))
    ind_loss.append(cur_loss)
ind_loss = tf.transpose(tf.stack(ind_loss), [1, 0])

## training and test predictions

self.restore(sess,True)
self.feed_dict[self.ph['phase_train_mdn']] = False

all_tt = []
all_locs = []
all_preds = []
for train_step in range(len(train_data)):
    data_ndx = train_step % len(m_train_data)
    cur_bpred = m_train_data[data_ndx][0]
    self.feed_dict[self.ph['base_locs']] = \
        PoseTools.getBasePredLocs(cur_bpred, self.conf)
    self.feed_dict[self.ph['base_pred']] = cur_bpred
    cur_te_loss, cur_ind_loss, pred_weights, pred_means, pred_std = \
        sess.run([self.loss, ind_loss, tf.nn.softmax(self.mdn_logits, dim=1),
                  self.mdn_locs, self.mdn_scales], feed_dict=self.feed_dict)
    val_loss = cur_te_loss
    jj = np.argmax(pred_weights, axis=1)
    f_pred = np.zeros([16, 5, 2])
    for a_ndx in range(16):
        for b_ndx in range(5):
            f_pred[a_ndx, b_ndx, :] = pred_means[a_ndx, jj[a_ndx, b_ndx],
                                      b_ndx, :]

    tt1 = np.sqrt(((f_pred * 4 - self.feed_dict[self.ph['base_locs']]) ** 2).sum(axis=2))
    all_tt.append(tt1)
    all_locs.append(self.feed_dict[self.ph['base_locs']])
    all_preds.append([pred_weights, pred_means, pred_std, val_loss, cur_ind_loss])
    if train_step % 10 == 0:
        print('{},{}'.format(train_step, len(train_data)))

train_t = np.array(all_tt).reshape([-1, 5])
train_w = np.array([i[0] for i in all_preds]).reshape([-1, 256, 5])
train_m = np.array([i[1] for i in all_preds]).reshape([-1, 256, 5, 2])
train_s = np.array([i[2] for i in all_preds]).reshape([-1, 256, 5, 2])
train_l = np.array(all_locs).reshape([-1, 5, 2])
train_loss = np.array([i[4] for i in all_preds]).reshape([-1, 5])
train_lm = (train_l / 4) % 16

with open('mdn_train_preds','wb') as f:
    pickle.dump( [train_t,train_w,train_m,train_s,
                  train_l,train_loss,train_lm],f)


all_tt = []
all_locs = []
all_preds = []
for train_step in range(len(m_test_data)):
    data_ndx = train_step % len(m_test_data)
    cur_bpred = m_test_data[data_ndx][0]
    self.feed_dict[self.ph['base_locs']] = \
        PoseTools.getBasePredLocs(cur_bpred, self.conf)
    self.feed_dict[self.ph['base_pred']] = cur_bpred
    cur_te_loss, cur_ind_loss, pred_weights, pred_means, pred_std = \
        sess.run([self.loss, ind_loss, tf.nn.softmax(self.mdn_logits, dim=1),
                  self.mdn_locs, self.mdn_scales], feed_dict=self.feed_dict)
    val_loss = cur_te_loss
    jj = np.argmax(pred_weights, axis=1)
    f_pred = np.zeros([16, 5, 2])
    for a_ndx in range(16):
        for b_ndx in range(5):
            f_pred[a_ndx, b_ndx, :] = pred_means[a_ndx, jj[a_ndx, b_ndx],
                                      b_ndx, :]

    tt1 = np.sqrt(((f_pred * 4 - self.feed_dict[self.ph['base_locs']]) ** 2).sum(axis=2))
    all_tt.append(tt1)
    all_locs.append(self.feed_dict[self.ph['base_locs']])
    all_preds.append([pred_weights, pred_means, pred_std, val_loss, cur_ind_loss])
    if train_step % 10 == 0:
        print('{},{}'.format(train_step, len(m_test_data)))

all_t = np.array(all_tt).reshape([-1, 5])
all_w = np.array([i[0] for i in all_preds]).reshape([-1, 256, 5])
all_m = np.array([i[1] for i in all_preds]).reshape([-1, 256, 5, 2])
all_s = np.array([i[2] for i in all_preds]).reshape([-1, 256, 5, 1])
all_l = np.array(all_locs).reshape([-1, 5, 2])
all_loss = np.array([i[4] for i in all_preds]).reshape([-1, 5])
all_lm = (all_l / 4) % 16

with open('mdn_test_preds','wb') as f:
    pickle.dump( [all_t,all_w,all_m,all_s,all_l,all_loss,all_lm],f)

sel_count = -1

##

with open('mdn_test_preds','rb') as f:
    [all_t, all_w, all_m, all_s, all_l, all_loss, all_lm] = pickle.load(f)

with open('mdn_train_preds','rb') as f:
    [train_t, train_w, train_m, train_s,
     train_l, train_loss, train_lm] = pickle.load(f)

sel_count = -1
##

sel_count += 1
zz = np.where(all_t.sum(axis=1) > 0)[0]

kk = np.arange(256) // 4
xx = kk // 8
yy = kk % 8

while True:
    # sel = np.random.choice(zz)
    sel = zz[sel_count]
    pp = np.where(all_t[sel, :] > 0)[0]
    if pp.size > 0:
        break
    else:
        sel_count += 1
sel_cls = np.random.choice(pp)
d_ndx = sel // conf.batch_size
i_ndx = sel % conf.batch_size

cur_l = all_l[sel, sel_cls, :] / 4 // 16

sel_ndx = np.where((xx == (cur_l[1])) & (yy == (cur_l[0])))[0]

diff_m = all_m[sel, sel_ndx, sel_cls, :] * 4 - all_l[sel, sel_cls, :]

mdn_pred_out = np.zeros([128, 128])
cur_m = all_m[sel, :, sel_cls, :]
cur_s = all_s[sel, :, sel_cls, :]
cur_w = all_w[sel, :, sel_cls]
for ndx in range(256):
    osz = 128
    if cur_w[ndx] < 0.02:
        continue
    cur_locs = cur_m[ndx:ndx + 1, :].astype('int')[np.newaxis, :, :]
    cur_scale = cur_s[ndx, :].mean().astype('int')
    cur_limg = (PoseTools.createLabelImages(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
    mdn_pred_out += cur_w[ndx] * cur_limg[0, ..., 0]
maxndx = np.argmax(mdn_pred_out)
curloc = np.array(np.unravel_index(maxndx,mdn_pred_out.shape))

f = plt.figure(figsize=[30,12])
# f, ax = plt.subplots(1, 3)
ax = []
for ndx in range(3):
    ax.append(f.add_subplot(1,3,ndx+1))
c_im = (m_test_data[d_ndx][0][i_ndx, :, :, sel_cls] + 1) / 2
c_im = np.clip(c_im, 0, 1)
i_im = m_test_data[d_ndx][3][i_ndx, 0, :, :]
ax[0].imshow(c_im, interpolation='nearest')
ax[0].scatter(all_l[sel, sel_cls, 0] / 4, all_l[sel, sel_cls, 1] / 4)
ax[0].scatter(all_m[sel, sel_ndx, sel_cls, 0], all_m[sel, sel_ndx, sel_cls, 1], c='r')
ax[1].imshow(mdn_pred_out, vmax=1., interpolation='nearest')
ax[1].scatter(all_l[sel, sel_cls, 0] / 4, all_l[sel, sel_cls, 1] / 4)
ax[1].scatter(all_m[sel, sel_ndx, sel_cls, 0], all_m[sel, sel_ndx, sel_cls, 1], c='r')
ax[1].set_title('{}'.format(sel))
ax[0].set_title('{:.2f},{},{}'.format(c_im.max(), sel_cls, sel))
ax[2].imshow(i_im, cmap='gray')
ax[2].scatter(all_l[sel, sel_cls, 0], all_l[sel, sel_cls, 1])
ax[2].scatter(curloc[1]*4,curloc[0]*4,c='r')
# ax[2].scatter(all_m[sel,sel_ndx,sel_cls,0]*4, all_m[sel,sel_ndx,sel_cls,1]*4,c='r')

for cur_ax in ax:
    cur_ax.set_xticks(np.arange(0, 128, 16));
    cur_ax.set_yticks(np.arange(0, 128, 16));

    cur_ax.set_xticklabels(np.arange(0, 128, 16));
    cur_ax.set_yticklabels(np.arange(0, 128, 16));
    # Gridlines based on minor ticks
    cur_ax.grid(which='major', color='w', linestyle='-', linewidth=1)

##
print('yes')

## ------------------------------------------------------ ##


#
# zz = np.where(all_t.sum(axis=1) > 40)[0]
#
# m_val = []
# for sel in zz:
#     sel_cls = np.argmax(all_t[sel, :])
#     d_ndx = sel // conf.batch_size
#     i_ndx = sel % conf.batch_size
#     c_im = m_train_data[d_ndx][0][i_ndx, :, :, sel_cls]
#     m_val.append(c_im.max())
#
# plt.scatter(all_t.sum(axis=1)[zz], m_val)
#
# ## select a training example with bad error
#
# zz = np.where(all_t.sum(axis=1) > 40)[0]
#
# kk = np.arange(256) // 4
# xx = kk // 8
# yy = kk % 8
#
# while True:
#     # sel = np.random.choice(zz)
#     sel = zz[sel_count]
#     pp = np.where(all_t[sel, :] > 20)[0]
#     if pp.size > 0:
#         break
#     else:
#         sel_count += 1
# sel_cls = np.random.choice(pp)
# d_ndx = sel // conf.batch_size
# i_ndx = sel % conf.batch_size
#
# cur_l = all_l[sel, sel_cls, :] / 4 // 16
#
# sel_ndx = np.where((xx == (cur_l[0])) & (yy == (cur_l[1])))[0]
#
# diff_m = all_m[sel, sel_ndx, sel_cls, :] * 4 - all_l[sel, sel_cls, :]
#
# mdn_pred_out = np.zeros([128, 128])
# cur_m = all_m[sel, :, sel_cls, :]
# cur_s = all_s[sel, :, sel_cls, :]
# cur_w = all_w[sel, :, sel_cls]
# for ndx in range(256):
#     osz = 128
#     if cur_w[ndx] < 0.02:
#         continue
#     cur_locs = cur_m[ndx:ndx + 1, :].astype('int')[np.newaxis, :, :]
#     cur_scale = cur_s[ndx, :].mean().astype('int')
#     cur_limg = (PoseTools.createLabelImages(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
#     mdn_pred_out += cur_w[ndx] * cur_limg[0, ..., 0]
#
# f, ax = plt.subplots(1, 3)
# c_im = (m_train_data[d_ndx][0][i_ndx, :, :, sel_cls] + 1) / 2
# c_im = np.clip(c_im, 0, 1)
# i_im = m_train_data[d_ndx][3][i_ndx, 0, :, :]
# ax[0].imshow(c_im, interpolation='nearest')
# ax[0].scatter(all_l[sel, sel_cls, 0] / 4, all_l[sel, sel_cls, 1] / 4)
# ax[0].scatter(all_m[sel, sel_ndx, sel_cls, 0], all_m[sel, sel_ndx, sel_cls, 1], c='r')
# ax[1].imshow(mdn_pred_out, vmax=1., interpolation='nearest')
# ax[1].scatter(all_l[sel, sel_cls, 0] / 4, all_l[sel, sel_cls, 1] / 4)
# ax[1].scatter(all_m[sel, sel_ndx, sel_cls, 0], all_m[sel, sel_ndx, sel_cls, 1], c='r')
# ax[0].set_title('{:.2f},{},{}'.format(c_im.max(), sel_cls, sel))
# ax[2].imshow(i_im, cmap='gray')
# ax[2].scatter(all_l[sel, sel_cls, 0], all_l[sel, sel_cls, 1])
# # ax[2].scatter(all_m[sel,sel_ndx,sel_cls,0]*4, all_m[sel,sel_ndx,sel_cls,1]*4,c='r')
#
# for cur_ax in ax:
#     cur_ax.set_xticks(np.arange(0, 128, 16));
#     cur_ax.set_yticks(np.arange(0, 128, 16));
#
#     cur_ax.set_xticklabels(np.arange(0, 128, 16));
#     cur_ax.set_yticklabels(np.arange(0, 128, 16));
#     # Gridlines based on minor ticks
#     cur_ax.grid(which='major', color='k', linestyle='-', linewidth=0.1)
#
# sel_count += 1
##

# vv = tf.global_variables()
# gg = tf.gradients(self.loss, vv)
# kk = [ndx for ndx, i in enumerate(gg) if i is not None]
# vv = [vv[i] for i in kk]
# gg = [gg[i] for i in kk]

self.restore(sess,True)

d_ndx = sel // conf.batch_size
i_ndx = sel % conf.batch_size
cur_in_img = np.tile(m_test_data[d_ndx][0][i_ndx, ...],
                     [conf.batch_size, 1, 1, 1])
self.feed_dict[self.ph['base_pred']] = cur_in_img
self.feed_dict[self.ph['base_locs']] = \
    PoseTools.getBasePredLocs(cur_in_img, self.conf)
self.feed_dict[self.ph['step']] = 10000
self.feed_dict[self.ph['phase_train_mdn']] = False

p_loss, b_ind_loss, b_w, b_m, b_s = \
    sess.run([self.loss, ind_loss, tf.nn.softmax(self.mdn_logits, dim=1),
              self.mdn_locs, self.mdn_scales], feed_dict=self.feed_dict)

for count in range(10):
    sess.run(self.opt, self.feed_dict)

self.feed_dict[self.ph['phase_train_mdn']] = False

p_loss, p_ind_loss, p_w, p_m, p_s = \
    sess.run([self.loss, ind_loss, tf.nn.softmax(self.mdn_logits, dim=1),
              self.mdn_locs, self.mdn_scales], feed_dict=self.feed_dict)

mdn_b_img = np.zeros([128, 128])
for m_ndx in range(b_m.shape[1]):
    if b_w[0, m_ndx, sel_cls] < 0.02:
        continue
    cur_locs = b_m[0:1, m_ndx:m_ndx + 1, sel_cls, :].astype('int')
    cur_scale = 3
    curl = (PoseTools.createLabelImages(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
    mdn_b_img += b_w[0, m_ndx, sel_cls] * curl[0, ..., 0]

mdn_p_img = np.zeros([128, 128])
for m_ndx in range(p_m.shape[1]):
    if p_w[0, m_ndx, sel_cls] < 0.02:
        continue
    cur_locs = p_m[0:1, m_ndx:m_ndx + 1, sel_cls, :].astype('int')
    cur_scale = 3
    curl = (PoseTools.createLabelImages(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
    mdn_p_img += p_w[0, m_ndx, sel_cls] * curl[0, ..., 0]


f, ax = plt.subplots(1, 4)
c_im = (m_test_data[d_ndx][0][i_ndx, :, :, sel_cls] + 1) / 2
c_im = np.clip(c_im, 0, 1)
i_im = m_test_data[d_ndx][3][i_ndx, 0, :, :]
ax[0].imshow(c_im, interpolation='nearest')
ax[0].scatter(all_l[sel, sel_cls, 0] / 4, all_l[sel, sel_cls, 1] / 4)
ax[0].scatter(all_m[sel, sel_ndx, sel_cls, 0], all_m[sel, sel_ndx, sel_cls, 1], c='r')
ax[0].set_title('{:.2f},{},{}'.format(c_im.max(), sel_cls, sel))
ax[1].imshow(mdn_b_img , vmax=1., interpolation='nearest')
ax[1].scatter(all_l[sel, sel_cls, 0] / 4, all_l[sel, sel_cls, 1] / 4)
ax[1].scatter(b_m[0, sel_ndx, sel_cls, 0], b_m[0, sel_ndx, sel_cls, 1], c='r')
ax[2].imshow(mdn_p_img, vmax=1., interpolation='nearest')
ax[2].scatter(all_l[sel, sel_cls, 0] / 4, all_l[sel, sel_cls, 1] / 4)
ax[2].scatter(p_m[0, sel_ndx, sel_cls, 0], p_m[0, sel_ndx, sel_cls, 1], c='r')
ax[3].imshow(i_im, cmap='gray')
ax[3].scatter(all_l[sel, sel_cls, 0], all_l[sel, sel_cls, 1])
# ax[2].scatter(all_m[sel,sel_ndx,sel_cls,0]*4, all_m[sel,sel_ndx,sel_cls,1]*4,c='r')

for cur_ax in ax:
    cur_ax.set_xticks(np.arange(0, 128, 16));
    cur_ax.set_yticks(np.arange(0, 128, 16));

    cur_ax.set_xticklabels(np.arange(0, 128, 16));
    cur_ax.set_yticklabels(np.arange(0, 128, 16));
    # Gridlines based on minor ticks
    cur_ax.grid(which='major', color='k', linestyle='-', linewidth=0.1)


##
mod_tt = []
mod_locs = []
mod_preds = []
for train_step in range(len(train_data)):
    data_ndx = train_step % len(m_train_data)
    cur_bpred = m_train_data[data_ndx][0]
    self.feed_dict[self.ph['base_locs']] = \
        PoseTools.getBasePredLocs(cur_bpred, self.conf)
    self.feed_dict[self.ph['base_pred']] = cur_bpred
    cur_te_loss, cur_ind_loss, pred_weights, pred_means, pred_std = \
        sess.run([self.loss, ind_loss, tf.nn.softmax(self.mdn_logits, dim=1),
                  self.mdn_locs, self.mdn_scales], feed_dict=self.feed_dict)
    val_loss = cur_te_loss
    jj = np.argmax(pred_weights, axis=1)
    f_pred = np.zeros([16, 5, 2])
    for a_ndx in range(16):
        for b_ndx in range(5):
            f_pred[a_ndx, b_ndx, :] = pred_means[a_ndx, jj[a_ndx, b_ndx],
                                      b_ndx, :]

    tt1 = np.sqrt(((f_pred * 4 - self.feed_dict[self.ph['base_locs']]) ** 2).sum(axis=2))
    mod_tt.append(tt1)
    mod_locs.append(self.feed_dict[self.ph['base_locs']])
    mod_preds.append([pred_weights, pred_means, pred_std, val_loss, cur_ind_loss])
    if train_step % 10 == 0:
        print('{},{}'.format(train_step, len(train_data)))

mod_t = np.array(mod_tt).reshape([-1, 5])
mod_w = np.array([i[0] for i in mod_preds]).reshape([-1, 256, 5])
mod_m = np.array([i[1] for i in mod_preds]).reshape([-1, 256, 5, 2])
mod_s = np.array([i[2] for i in mod_preds]).reshape([-1, 256, 5, 2])
mod_l = np.array(mod_locs).reshape([-1, 5, 2])
mod_loss = np.array([i[4] for i in mod_preds]).reshape([-1, 5])
mod_lm = (mod_l / 4) % 16

##

hh = train_loss - mod_loss
qq = np.flipud(np.argsort(hh[:, sel_cls]))

f, ax = plt.subplots(2, 5)
ax = ax.flatten()

for ndx in range(10):
    cur_ndx = qq[ndx]
    cur_d_ndx = cur_ndx // conf.batch_size
    cur_i_ndx = cur_ndx % conf.batch_size
    mdn_pred_out1 = np.zeros([128, 128])
    mdn_pred_out2 = np.zeros([128, 128])

    for m_ndx in range(mod_m.shape[1]):
        if mod_w[cur_ndx, m_ndx, sel_cls] < 0.02:
            continue
        cur_locs = mod_m[cur_ndx:cur_ndx + 1, m_ndx:m_ndx + 1, sel_cls, :].astype('int')
        cur_scale = 3
        curl = (PoseTools.createLabelImages(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
        mdn_pred_out1 += mod_w[cur_ndx, m_ndx, sel_cls] * curl[0, ..., 0]

    for m_ndx in range(train_m.shape[1]):
        if train_w[cur_ndx, m_ndx, sel_cls] < 0.02:
            continue
        cur_locs = train_m[cur_ndx:cur_ndx + 1, m_ndx:m_ndx + 1, sel_cls, :].astype('int')
        cur_scale = 3
        curl = (PoseTools.createLabelImages(cur_locs, [osz, osz], 1, cur_scale) + 1) / 2
        mdn_pred_out2 += train_w[cur_ndx, m_ndx, sel_cls] * curl[0, ..., 0]
    mdn_pred_out = np.zeros([128, 128, 3])
    mdn_pred_out[:, :, 0] = mdn_pred_out1
    mdn_pred_out[:, :, 1] = mdn_pred_out2
    cur_bpred = (m_train_data[cur_d_ndx][0][cur_i_ndx, :, :, sel_cls] + 1) / 2
    mdn_pred_out[:, :, 2] = cur_bpred
    mdn_pred_out = np.clip(mdn_pred_out, 0, 1)
    ax[ndx].imshow(mdn_pred_out, interpolation='nearest')




## ================= whole network =======================



from stephenHeadConfig import conf as conf
import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES']= ''
import PoseMDN
import localSetup
import PoseTools

if os.environ['CUDA_VISIBLE_DEVICES'] is '':
    print('!!!!!Not USING GPU!!!!!')

conf.cachedir = os.path.join(localSetup.bdir,'cacheHead_MDN')
conf.expname = 'head_dw_logits_1e2'
conf.psz = 8
self = PoseMDN.PoseMDN(conf)
restore = True
trainType = 0
full = True

self.conf.trange = self.conf.imsz[0] // 25

mdn_dropout = 1.
self.create_ph()
self.createFeedDict()
self.feed_dict[self.ph['keep_prob']] = mdn_dropout
self.feed_dict[self.ph['phase_train_base']] = False
self.feed_dict[self.ph['phase_train_mdn']] = True
self.feed_dict[self.ph['learning_rate']] = 0.
self.trainType = trainType

with tf.variable_scope('base'):
    super(self.__class__, self).createBaseNetwork(doBatch=True)

if full:
    l7_layer = self.baseLayers['conv7']
else:
    l7_layer = tf.stop_gradient(self.baseLayers['conv7'])
# if full:
#     l7_layer = self.basePred
# else:
#     l7_layer = tf.stop_gradient(self.basePred)
self.create_network_joint(l7_layer)
self.openDBs()
self.createBaseSaver()
self.create_saver()

y_label = self.ph['locs'] / self.conf.rescale / self.conf.pool_scale
self.mdn_label = y_label
# data_dict = {}
# for ndx in range(self.conf.n_classes):
#     data_dict[y[ndx]] = y_label[:,ndx,:]
# inference = mymap.MAP(data=data_dict)
# inference.initialize(var_list=PoseTools.getvars('mdn'))
# self.loss = inference.loss
self.loss = self.my_loss_joint()

starter_learning_rate = 0.0001
decay_steps = 12000 * 8 / self.conf.batch_size
learning_rate = tf.train.exponential_decay(
    starter_learning_rate, self.ph['step'], decay_steps, 0.9)
# decay_steps = 5000 / 8 * self.conf.batch_size
# learning_rate = tf.train.exponential_decay(
#     starter_learning_rate,self.ph['step'],decay_steps, 0.1,
#     staircase=True)

mdn_steps = 50000 * 8 / self.conf.batch_size

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    self.opt = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(self.loss)

sess = tf.InteractiveSession()
self.createCursors(sess)
self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
self.feed_dict[self.ph['base_locs']] = np.zeros([self.conf.batch_size,
                                                 self.conf.n_classes, 2])
sess.run(tf.global_variables_initializer())
self.restore_base_full(sess,True)
self.restore(sess, restore)
l7_shape = self.baseLayers['conv7'].get_shape().as_list()
l7_shape[0] = self.conf.batch_size

self.feed_dict[self.ph['step']] = 0
self.feed_dict[self.ph['phase_train_mdn']] = False
self.feed_dict[self.ph['phase_train_base']] = False

ind_loss = []
for cls in range(self.conf.n_classes):
    ll = tf.nn.softmax(self.mdn_logits, dim=1)

    cur_scales = self.mdn_scales[:, :, cls]
    pp = self.mdn_label[:, cls:cls + 1, :]
    kk = tf.reduce_sum(tf.square(pp - self.mdn_locs[:, :, cls, :]), axis=2)
    cur_comp = tf.div(tf.exp(-kk / (cur_scales ** 2) / 2), 2 * np.pi * (cur_scales ** 2))

    pp = cur_comp * ll
    cur_loss = tf.log(tf.reduce_sum(pp, axis=1))
    ind_loss.append(cur_loss)
ind_loss = tf.transpose(tf.stack(ind_loss), [1, 0])

#
mod_tt = []
mod_locs = []
mod_preds = []
mod_x = []
mod_bpred = []

val_file = os.path.join(conf.cachedir, conf.valfilename + '.tfrecords')
c = 0
for record in tf.python_io.tf_record_iterator(val_file):
    c += 1

for step in range(c/conf.batch_size):
    self.updateFeedDict(self.DBType.Val, sess=sess, distort=False)
    self.feed_dict[self.ph['keep_prob']] = 1

    self.restoreBase(sess,True)
    b_pred = sess.run(self.basePred, feed_dict=self.feed_dict)
    self.restore_base_full(sess,True)
    self.restore(sess, restore)
    cur_te_loss, cur_ind_loss, pred_weights, pred_means, pred_std, raw_pred_weights = \
        sess.run([self.loss, ind_loss, tf.nn.softmax(self.mdn_logits, dim=1),
                  self.mdn_locs, self.mdn_scales, self.mdn_logits], feed_dict=self.feed_dict)
    val_loss = cur_te_loss
    jj = np.argmax(pred_weights, axis=1)
    f_pred = np.zeros([conf.batch_size, conf.n_classes, 2])
    for a_ndx in range(conf.batch_size):
        for b_ndx in range(conf.n_classes):
            f_pred[a_ndx, b_ndx, :] = pred_means[a_ndx, jj[a_ndx],
                                      b_ndx, :]

    tt1 = np.sqrt(((f_pred * 4 - self.feed_dict[self.ph['locs']]) ** 2).sum(axis=2))
    mod_tt.append(tt1)
    mod_locs.append(self.feed_dict[self.ph['locs']])
    mod_preds.append([pred_weights, pred_means, pred_std, val_loss, cur_ind_loss, raw_pred_weights])
    mod_x.append(self.xs)
    mod_bpred.append(b_pred)
    if step % 10 == 0:
        print('{},{}'.format(step, 10))

mod_t = np.array(mod_tt).reshape([-1, 5])
mod_w = np.array([i[0] for i in mod_preds]).reshape([-1, 256])
mod_rw = np.array([i[5] for i in mod_preds]).reshape([-1, 256])
mod_m = np.array([i[1] for i in mod_preds]).reshape([-1, 256, 5, 2])
mod_s = np.array([i[2] for i in mod_preds]).reshape([-1, 256, 5])
mod_l = np.array(mod_locs).reshape([-1, 5, 2])
mod_loss = np.array([i[4] for i in mod_preds]).reshape([-1, 5])
mod_lm = (mod_l / 4) % 16
mod_x = np.array(mod_x).reshape([-1, 512, 512])
mod_bpred = np.array(mod_bpred).reshape([-1, 128, 128, 5])
gg = PoseTools.get_base_error(mod_l, mod_bpred, conf)
mod_ot = np.sqrt(np.sum(gg**2,2))

print(np.argmax(mod_t,axis=0))
print(np.max(mod_t,axis=0))
blocs = PoseTools.get_base_pred_locs(mod_bpred, conf)
sc = -1
##
sc += 1
kk = np.flipud(np.argsort(mod_t.flatten()))
[yy,xx] = np.unravel_index(kk[:30],mod_t.shape)
# kk = np.argsort(mod_t.sum(axis=1))
# yy = kk[:10]
print(np.unique(yy))

sel = np.unique(yy)[sc]
print(mod_t[sel,:])

f = plt.figure(figsize=[12,12])
ax = f.add_subplot(111)
plt.imshow(mod_x[sel,:,:],cmap='gray')
plt.scatter(mod_l[sel,:,0], mod_l[sel,:,1],s=20)
w_sort = np.flipud(np.argsort(mod_w[sel,...]))
jx = w_sort[0]
plt.scatter(mod_m[sel,jx,:,0]*4,mod_m[sel,jx,:,1]*4,c='r')
# plt.scatter(blocs[sel,:,0],blocs[sel,:,1],c='g')
jx1 = w_sort[1]
plt.scatter(mod_m[sel,jx1,:,0]*4,mod_m[sel,jx1,:,1]*4,c='w')
tt = mod_m[sel,...]
zz = np.sqrt(np.sum((tt*4-mod_l[sel,...])**2,axis=(1,2)))
jx2 = zz.argmin()
plt.scatter(mod_m[sel,jx2,:,0]*4,mod_m[sel,jx2,:,1]*4,c='c')
ax.set_title('{}, {:.2f}, {:.2f}, {:.2f}'.format(sel,mod_w[sel,jx],mod_w[sel,jx1],mod_w[sel,jx2]))

##
import cv2
def createPredImage(pred_scores, n_classes):
    im = np.zeros(pred_scores.shape[0:2] + (3,))
    im[:,:,0] = np.argmax(pred_scores, 2).astype('float32') / (n_classes) * 180
    im[:,:,1] = (np.max(pred_scores, 2) + 1) / 2 * 255
    im[:,:,2] = 255.
    im = np.clip(im,0,255)
    im = im.astype('uint8')
    return cv2.cvtColor(im,cv2.COLOR_HSV2RGB)


sel = 1
ll = np.where(mod_w[sel,:]>0.2)[0]
f = plt.figure()
ax = f.add_subplot(111)
ax.imshow(mod_x[sel,:,:],cmap='gray',vmax=255)
for curl in ll:
    ax.scatter(mod_m[sel,curl,:,0]*4,mod_m[sel,curl,:,1]*4)

plt.figure()
from scipy import misc
pimg = createPredImage(mod_bpred[sel,...],5)
pimg = misc.imresize(pimg,4.)
hh1 = cv2.cvtColor(pimg,cv2.COLOR_RGB2HSV)
hh = cv2.cvtColor(np.tile(mod_x[sel,:,:,np.newaxis],[1,1,3]),cv2.COLOR_RGB2HSV)
kk = hh
kk[:,:,0] = hh1[:,:,0]
kk[:,:,1] = np.maximum(hh1[:,:,1],hh[:,:,1])
rr = cv2.cvtColor(kk,cv2.COLOR_HSV2RGB)
plt.imshow(rr)
plt.savefig('/groups/branson/home/kabram/temp/mdn_plots/part_detector_hmap.png',dpi=240)
plt.figure()
plt.imshow(mod_x[sel,:,:],cmap='gray',vmax=255)
plt.scatter(mod_l[sel,:,0],mod_l[sel,:,1],c='r',vmax=255)
plt.savefig('/groups/branson/home/kabram/temp/mdn_plots/orig_img.png',dpi=240)

##
f = plt.figure()
ax = f.add_subplot(111)
ll = np.zeros([800,256,6,2])
ll[:,:,0,:] = mod_m[:,:,0,:]
ll[:,:,1,:] = mod_m[:,:,2,:]
ll[:,:,2,:] = mod_m[:,:,3,:]
ll[:,:,3,:] = mod_m[:,:,1,:]
ll[:,:,4,:] = mod_m[:,:,4,:]
ll[:,:,5,:] = mod_m[:,:,0,:]
for x in range(8):
    for y in range(8):
        f.clf()
        ax = f.add_subplot(111)
        cur_idx = np.arange(4*(8*y+x),4*(8*y+x+1))
        ax.plot(ll[sel,cur_idx,:,0].T*4,ll[sel,cur_idx,:,1].T*4,c='r')
        sz = 208
        x_s = 64*x-sz; x_e = 64*x+sz
        y_s = 64*y-sz; y_e = 64*y+sz
        ax.plot([x_s,x_s,x_e,x_e,x_s],[y_s,y_e,y_e,y_s,y_s],c='r')
        ax.imshow(mod_x[sel,...],cmap='gray',vmax=255)
        plt.pause(0.5)
        # f.savefig('/groups/branson/home/kabram/temp/mdn_plots/ex_x{}_y{}.jpg'.format(x,y),
        #           dpi=240)
        # ax.set_xlim([0,512]); ax.set_ylim([0,512])



##

mdn_pred_out = np.zeros([128, 128, conf.n_classes])
for cls in range(conf.n_classes):
    for ndx in range(pred_means.shape[1]):
        if mod_w[sel,ndx] < 0.02:
            continue
        cur_locs = mod_m[sel:sel + 1, ndx:ndx + 1, cls, :].astype('int')
        # cur_scale = pred_std[sel, ndx, cls, :].mean().astype('int')
        cur_scale = mod_s[sel, ndx, cls].astype('int')
        curl = (PoseTools.create_label_images(cur_locs, [128, 128], 1, cur_scale) + 1) / 2
        mdn_pred_out[:, :, cls] += mod_w[sel, ndx] * curl[0, ..., 0]

mdn_pred_out = mdn_pred_out*2-1
plt.figure()
from scipy import misc
pimg = createPredImage(mdn_pred_out,5)
pimg = misc.imresize(pimg,4.)
hh1 = cv2.cvtColor(pimg,cv2.COLOR_RGB2HSV)
hh = cv2.cvtColor(np.tile(mod_x[sel,:,:,np.newaxis],[1,1,3]),cv2.COLOR_RGB2HSV)
kk = hh
kk[:,:,0] = hh1[:,:,0]
kk[:,:,1] = np.maximum(hh1[:,:,1],hh[:,:,1])
rr = cv2.cvtColor(kk,cv2.COLOR_HSV2RGB)
plt.imshow(rr)
plt.savefig('/groups/branson/home/kabram/temp/mdn_plots/part_detector_mdn_hmap.png',dpi=240)



##
