from Pose_mdn_joint import Pose_mdn_joint
import tensorflow
vv = [int(v) for v in tensorflow.__version__.split('.')]
if (vv[0]==1 and vv[1]>12) or vv[0]==2:
    tf = tensorflow.compat.v1
else:
    tf = tensorflow
import numpy as np


class Pose_multi_mdn_joint(Pose_mdn_joint):

    def __init__(self,conf, name='deepnet',**kwargs):
        Pose_mdn_joint.__init__(self,conf,name=name,is_multi=True,**kwargs)


    def set_shape(self):
        im, locs, info, hmap = self.inputs
        conf = self.conf
        in_sz = [int(sz//conf.rescale) for sz in conf.imsz]
        im.set_shape([conf.batch_size,
                      in_sz[0] + self.pad_y,
                      in_sz[1] + self.pad_x,
                      conf.img_dim])
        hmap.set_shape([conf.batch_size, in_sz[0], in_sz[1],conf.n_classes])
        locs.set_shape([conf.batch_size,conf.max_n_animals, conf.n_classes,2])
        info.set_shape([conf.batch_size,3])

    def l2_loss(self,X,y):

        assert self.k_joint==1, 'This only works for k_joint ==1'
        locs_offset = self.offset
        n_x = self.n_x_j
        n_y = self.n_y_j
        n_classes = self.conf.n_classes
        n_max = self.conf.max_n_animals

        mdn_locs_joint, mdn_locs, mdn_logits_joint, mdn_logits = X
        logits_all = tf.reshape(mdn_logits_joint,[-1,n_x*n_y,1])
        ll_joint = tf.nn.sigmoid(logits_all)
        # ll_joint is the weight for each prediction and with sigmoid it is always positiive. Shape is  B x P x 1
        self.sigmoid_logits = ll_joint

        mdn_locs_joint = tf.reshape(mdn_locs_joint,[-1,n_x*n_y,1,n_classes, 2])
        n_preds_joint = n_x*n_y
        cur_loss = 0
        all_pp = []

        in_locs = tf.expand_dims(y,1)
        # to match the dimensions of predictions. Shape is B x 1 X L x N x 2

        valid = tf.reduce_any(y>-10000, axis=[-1, -2])
        # valid keeps track of how many labels are for for valid animals. Shape is B x 1 x L

        is_occ = in_locs<-10000
        # is_occ keeps track of labels that are NaN. Shape is B x 1 x L x N
        is_occ = tf.tile(is_occ,[1,n_x*n_y,1,1,1])

        mdn_locs_joint_s = mdn_locs_joint*self.offset

        dd = tf.sqrt(tf.reduce_sum((mdn_locs_joint_s-in_locs)**2,-1))
        # dd = tf.where(is_occ,tf.ones_like(dd)*1000,dd)
        # dd has the distance between all the P predictions and all the L labels. Size is B x P x L x N. Zero all occluded landmarks
        dd = tf.where(is_occ[...,0],tf.ones_like(dd)*1000,dd)
        self.dd = dd

        # Sum the distances across all landmarks. Shape is B x P x L
        dd_all = tf.reduce_sum(dd,-1)
        self.dd_all = dd_all  # Shittily keep track of each tensor for debugging. augh.

        soft_noise = tf.random.uniform(dd_all.get_shape(),minval=0,maxval=0.0001)
        # a bit of noise so that softmax doesnt fail for invalid labels
        # Do softmax for all the L distances of a prediction, and that will be the soft assignment of that prediction to the L labels. The shape is B x P x L
        p_assign = tf.nn.softmax(-tf.stop_gradient(dd_all)+soft_noise,axis=2)
        # p_assign = tf.where(tf.tile(valid[...,0,0],[1,n_x*n_y,1]),p_assign,tf.zeros_like(p_assign))
        self.p_assign = p_assign

        p_sum = tf.reduce_sum(p_assign*ll_joint,1,keepdims=True)
        p_sum = tf.where(tf.expand_dims(valid,1),p_sum,tf.ones_like(p_sum))
        p_norm = (p_assign*ll_joint)/(p_sum + 0.000001)
        self.p_norm = p_norm

        # Each prediction should be close to its assigned label.
        aa = dd_all*p_norm
        # aa = tf.where(tf.is_nan(aa),tf.zeros_like(aa),aa)
        loss_pred = tf.reduce_sum(aa)

        self.loss_pred = aa

        # weigh the assignment by weights (logits_joint) to get weighted assignment
        w_assign = ll_joint * p_assign

        # now, the total weight that gets attached to each valid label should sum to 1.
        w_label = tf.reduce_sum(w_assign,1)
        loss_label = tf.reduce_sum((w_label-tf.cast(valid,tf.float32))**2)
        loss_label = loss_label * self.conf.n_classes
        self.w_assign = w_assign
        self.valid = valid
        self.loss_label = loss_label

        joint_loss = loss_label + loss_pred

        ## refinement
        n_x = self.n_x_r
        n_y = self.n_y_r
        ll_img = tf.reshape(mdn_logits, [-1, n_x * n_y, self.k_ref, n_classes])
        cur_loss = 0
        all_pp = []
        locs_noise = self.conf.get('mdn_joint_ref_noise',1.)
        if locs_noise > 0.001:
            mdn_locs_noise = mdn_locs_joint + tf.random.uniform(mdn_locs_joint.get_shape(),minval=-locs_noise,maxval=locs_noise)
        else:
            mdn_locs_noise = mdn_locs_joint

        mdn_locs_noise = mdn_locs_noise*self.ref_scale
        for b in range(self.conf.batch_size):
            for g in range(n_max):
                cur_pp = []
                selex = tf.argmax(w_assign[b, :, g])
                for cls in range(self.conf.n_classes):
                    idx_y = tf.cast(tf.clip_by_value(tf.round(mdn_locs_noise[b,selex,0,cls,1]),0,n_y-1),tf.int64)
                    idx_x = tf.cast(tf.clip_by_value(tf.round(mdn_locs_noise[b,selex,0,cls,0]),0,n_x-1),tf.int64)
                    # ids are predicted as x,y to match input locs.
                    pp = y[b, g, cls:cls + 1, :] 
                    occ_pts = tf.is_finite(pp) & (y[b,g,cls:cls+1,:] > -10000)
                    # pp = tf.where(occ_pts, pp, tf.zeros_like(pp))
                    occ_pts_pred = tf.tile(occ_pts, [self.k_ref, 1])
                    qq = mdn_locs[b, idx_y,idx_x, :, cls, :] * locs_offset/self.ref_scale
                    # qq = tf.where(occ_pts_pred, qq, tf.zeros_like(qq))
                    kk = tf.sqrt(tf.reduce_sum(tf.square(pp - qq), axis=1))
                    # kk is the distance between all predictions at location selex for point cls
                    kk = tf.where(occ_pts_pred[...,0],kk,tf.zeros_like(kk))
                    ll = mdn_logits[b,idx_y,idx_x,:,cls]
                    ll = tf.nn.softmax(ll)
                    pp = ll * kk
                    cur_loss += tf.reduce_sum(pp)
                    cur_pp.append(pp)
                all_pp.append(cur_pp)

        self.ref_loss_pp = all_pp
        ref_loss = cur_loss

        tot_loss = joint_loss + ref_loss
        return tot_loss / self.conf.n_classes


    def get_joint_pred(self,preds):
        n_max = self.conf.max_n_animals
        locs_joint, locs_ref, logits_joint, logits_ref = preds
        bsz = locs_joint.shape[0]
        n_classes = locs_joint.shape[-2]
        n_x_j = self.n_x_j; n_y_j = self.n_y_j
        n_x_r = self.n_x_r; n_y_r = self.n_y_r
        locs_offset = self.offset
        k_ref = self.k_ref
        ll_joint_img = np.reshape(logits_joint,[-1,n_x_j*n_y_j])
        ll_img = np.reshape(logits_ref,[-1,n_x_r*n_y_r,k_ref,n_classes])
        locs_ref = locs_ref * locs_offset / self.ref_scale

        preds_ref = np.ones([bsz,n_max, n_classes,2]) * np.nan
        preds_joint = np.ones([bsz,n_max, n_classes,2]) * np.nan

        for ndx in range(bsz):
            n_preds = np.count_nonzero(ll_joint_img[ndx,:]>0)
            n_preds = n_max if n_preds>n_max else n_preds
            n_preds = 1 if n_preds < 1 else n_preds
            ids = np.argsort(-ll_joint_img[ndx,:])
            for cur_n in range(n_preds):
                sel_ex = ids[cur_n]
                idx = np.unravel_index(sel_ex, [n_y_j, n_x_j])
                preds_joint[ndx,cur_n,...] = locs_joint[ndx,idx[0],idx[1],...] * locs_offset
                for cls in range(n_classes):
                    mm = np.round(locs_joint[ndx,idx[0],idx[1],cls,:]).astype('int')*self.ref_scale
                    mm_y = np.clip(mm[1],0,n_y_r-1)
                    mm_x = np.clip(mm[0],0,n_x_r-1)
                    pt_selex = np.argmax(logits_ref[ndx,mm_y,mm_x,:,cls])
                    cur_pred = locs_ref[ndx,mm_y,mm_x,pt_selex,cls,:]
                    preds_ref[ndx,cur_n,cls,:] = cur_pred
        return preds_ref,preds_joint

    def compute_dist(self, preds, locs):
        locs = locs.copy()
        is_valid_nan = np.any(np.invert(np.isnan(locs)),(-1,-2))
        is_valid_low = np.any(locs>-10000,(-1,-2))
        is_valid = is_valid_low & is_valid_nan
        pp = self.get_joint_pred(preds)
        dd_all = np.linalg.norm(locs[:,np.newaxis,...]-pp[:,:,np.newaxis,...],axis=-1)
        dd = np.nanmean(dd_all,axis=-1)
        mm = np.nanmin(dd,axis=1) # closest prediction to each label
        mean_d = np.mean(mm[is_valid])

        return mean_d
