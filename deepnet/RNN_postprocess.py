import PoseTools
import multiResData
import tensorflow as tf
import PoseUNet_resnet
import os
import h5py
import numpy as np
import movies
from scipy import io as sio
from multiResData import int64_feature, float_feature, bytes_feature
import pickle
import sys


class RNN_pp(object):


    def __init__(self, conf, mdn_name):
        self.conf = conf
        self.mdn_name = mdn_name
        self.rnn_pp_hist = 128
        self.train_rep = 2
        self.conf.check_bounds_distort = False

    def create_db(self, split_file=None):
        assert  self.rnn_pp_hist % self.conf.batch_size == 0, 'make sure the history is a multiple of batch size'
        assert len(self.conf.mdn_groups)==1, 'This works only for single group. check for line 118'
        net = PoseUNet_resnet.PoseUMDN_resnet(self.conf,self.mdn_name)
        sess, _ = net.restore_net_common(net.create_network)

        conf = self.conf
        on_gt = False
        db_files = ()
        if split_file is not None:
            predefined = PoseTools.json_load(split_file)
            split = False
        else:
            predefined = None

        mov_split = None

        local_dirs, _ = multiResData.find_local_dirs(conf, on_gt=False)
        lbl = h5py.File(conf.labelfile, 'r')
        view = conf.view
        flipud = conf.flipud
        npts_per_view = np.array(lbl['cfg']['NumLabelPoints'])[0, 0]
        sel_pts = int(view * npts_per_view) + conf.selpts

        out_fns = [True, False]
        data = [[],[]]
        count = 0
        for ndx, dir_name in enumerate(local_dirs):

            cur_pts = multiResData.trx_pts(lbl, ndx, on_gt)
            crop_loc = PoseTools.get_crop_loc(lbl, ndx, view, on_gt)
            cap = movies.Movie(dir_name)

            if conf.has_trx_file:
                trx_files = multiResData.get_trx_files(lbl, local_dirs, on_gt)
                trx = sio.loadmat(trx_files[ndx])['trx'][0]
                n_trx = len(trx)
                trx_split = np.random.random(n_trx) < conf.valratio
            else:
                trx = [None]
                n_trx = 1
                trx_split = None
                cur_pts = cur_pts[np.newaxis, ...]

            for trx_ndx in range(n_trx):

                frames = multiResData.get_labeled_frames(lbl, ndx, trx_ndx, on_gt)
                cur_trx = trx[trx_ndx]
                for fnum in frames:
                    info = [ndx, fnum, trx_ndx]
                    cur_out = multiResData.get_cur_env(out_fns, split, conf, info, mov_split, trx_split=trx_split, predefined=predefined)
                    num_rep = 1 + cur_out*(self.train_rep-1)

                    for rep in range(num_rep):
                        cur_pred = np.ones([self.rnn_pp_hist,self.conf.n_classes,2])
                        cur_ims = []
                        cur_labels = []
                        for fndx in reversed(range(self.rnn_pp_hist)):
                            frame_in, cur_loc = multiResData.get_patch( cap, fnum, conf, cur_pts[trx_ndx, fnum, :, sel_pts], cur_trx=cur_trx, flipud=flipud, crop_loc=crop_loc, offset=-fndx)
                            cur_labels.append(cur_loc)
                            cur_ims.append(frame_in)

                        cur_ims = np.array(cur_ims)
                        cur_labels = np.array(cur_labels)

                        cur_ims, cur_labels = PoseTools.preprocess_ims(cur_ims, cur_labels, conf, distort=cur_out,scale= self.conf.rescale,group_sz=self.rnn_pp_hist)

                        bsize = self.conf.batch_size
                        nbatches = self.rnn_pp_hist/bsize
                        for bndx in range(nbatches):
                            start = bndx*bsize
                            end = (bndx+1)*bsize
                            net.fd[net.inputs[0]] = cur_ims[start:end,...]
                            net.fd[net.inputs[1]] = cur_labels[start:end,...]
                            info_fd = np.zeros([bsize,3])
                            info_fd[:,0] = ndx; info_fd[:,1] = np.arange(start,end); info_fd[:,2] = trx_ndx
                            net.fd[net.inputs[2]] = info_fd
                            net.fd[net.inputs[3]] = np.zeros(net.inputs[3]._shape_as_list())

                            cur_m, cur_s, cur_w = sess.run(net.pred, net.fd)
                            cur_w = cur_w[:,:,0]
                            nx = np.argmax(cur_w, axis=1)
                            cur_pred[start:end,:,:] = cur_m[np.arange(bsize),nx,:,:]

                        cur_info = [ndx, fnum, trx_ndx]
                        if cur_out:
                            data[0].append([cur_pred, cur_labels[-1,...], cur_info])
                        else:
                            data[1].append([cur_pred, cur_labels[-1,...], cur_info])
                        count += 1

                    if count % 50 == 0:
                        sys.stdout.write('.')
                    if count % 2000 == 0:
                        sys.stdout.write('\n')

            cap.close()  # close the movie handles

        lbl.close()

        with open(os.path.join(conf.cachedir,'rnn_pp.p')) as f:
            pickle.dump(data,f)


