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
import time

class RNN_pp(object):


    def __init__(self, conf, mdn_name, name ='rnn_pp'):

        self.conf = conf
        self.mdn_name = mdn_name
        self.rnn_pp_hist = 128
        self.train_rep = 2
        self.conf.check_bounds_distort = False
        self.ckpt_file = os.path.join( conf.cachedir, conf.expname + '_' + name + '_ckpt')
        self.name = name


    def create_db(self, split_file=None):
        assert  self.rnn_pp_hist % self.conf.batch_size == 0, 'make sure the history is a multiple of batch size'
        assert len(self.conf.mdn_groups)==1, 'This works only for single group. check for line 118'
        net = PoseUNet_resnet.PoseUMDN_resnet(self.conf,self.mdn_name)
        sess, _ = net.restore_net_common(net.create_network)

        conf = self.conf
        on_gt = False
        db_files = ()
        if split_file is not None:
            self.conf.splitType = 'predefined'
            predefined = PoseTools.json_load(split_file)
            split = True
        else:
            predefined = None
            split = False

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


    def get_var_list(self):
        var_list = tf.global_variables()
        return var_list


    def create_saver(self):
        saver = {}
        name = self.name
        saver['out_file'] = os.path.join(
            self.conf.cachedir,
            self.conf.expname + '_' + name)
        saver['train_data_file'] = os.path.join(
                self.conf.cachedir,
                self.conf.expname + '_' + name + '_traindata')
        saver['ckpt_file'] = self.ckpt_file
        var_list = self.get_var_list()
        saver['saver'] = (tf.train.Saver(var_list=var_list,
                                         max_to_keep=self.conf.maxckpt,
                                         save_relative_paths=True))
        self.saver = saver

    def restore(self, sess, model_file=None):
        saver = self.saver
        if model_file is not None:
            latest_model_file = model_file
            saver['saver'].restore(sess, model_file)
        else:
            grr = os.path.split(self.ckpt_file) # angry that get_checkpoint_state doesnt accept complete path to ckpt file. Damn idiots!
            latest_ckpt = tf.train.get_checkpoint_state(grr[0],grr[1])
            latest_model_file = latest_ckpt.model_checkpoint_path
            saver['saver'].restore(sess, latest_model_file)
        return latest_model_file


    def save(self, sess, step):
        saver = self.saver
        out_file = saver['out_file'].replace('\\', '/')
        saver['saver'].save(sess, out_file, global_step=step,
                            latest_filename=os.path.basename(saver['ckpt_file']))
        print('Saved state to %s-%d' % (out_file, step))


    def create_network(self):
        lstm_size = 256
        batch_size = self.conf.batch_size*2

        input_ph = tf.placeholder(tf.float32, [self.rnn_pp_hist, batch_size, self.conf.n_classes*2])
        out_ph = tf.placeholder(tf.float32, [batch_size, self.conf.n_classes*2])
        lr_ph = tf.placeholder(tf.float32)
        self.ph = {'input':input_ph, 'output':out_ph, 'learning_rate': lr_ph}

        in_w = tf.get_variable('in_weights',[self.conf.n_classes*2, lstm_size],initializer=tf.contrib.layers.xavier_initializer())
        in_b = tf.get_variable('softmax_weights',[lstm_size],initializer=tf.constant_initializer(0.))
        input = tf.nn.relu(tf.matmul(input_ph, in_w) + in_b)

        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        # Initial state of the LSTM memory.
        state = lstm.zero_state(batch_size, dtype=tf.float32)
        for cur_input in input:
            output, state = lstm(cur_input, state)

        softmax_w = tf.get_variable('softmax_weights',[lstm_size,self.conf.n_classes*2],initializer=tf.contrib.layers.xavier_initializer())
        softmax_b = tf.get_variable('softmax_weights',[self.conf.n_classes*2],initializer=tf.constant_initializer(0.))
        out = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.l2_loss(out-out_ph)

        self.out = out
        self.cost = loss

        self.fd = {}
        for k in self.ph.keys():
            self.fd[self.ph[k]] = np.zeros(self.ph[k]._shape_as_list())


    def train_step(self, step, sess, learning_rate, training_iters):
        cur_step = float(step)

        n_steps = self.conf.n_steps
        cur_lr = learning_rate * (self.conf.gamma ** (cur_step * n_steps / training_iters))
        self.fd[self.ph['learning_rate']] = cur_lr
        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
        sess.run(self.opt, self.fd, options=run_options)


    def create_optimizer(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.ph['learning_rate'])
            gradients, variables = zip(*optimizer.compute_gradients(self.cost))
            gradients = [None if gradient is None else
                         tf.clip_by_norm(gradient, 5.0)
                         for gradient in gradients]
            self.opt = optimizer.apply_gradients(zip(gradients, variables))


    def init_td(self):
        ex_td_fields = ['step']
        self.td_fields = ['loss','dist']
        for t_f in self.td_fields:
            ex_td_fields.append('train_' + t_f)
            ex_td_fields.append('val_' + t_f)
        train_info = {}
        for t_f in ex_td_fields:
            train_info[t_f] = []
        self.train_info = train_info

    def create_datasets(self):
        with open(self.conf.cachedir,'rnn_pp.p','r') as f:
            X = pickle.load(f)


    def fd_train(self):


    def fd_val(self):



    def compute_train_data(self, sess, db_type):
        self.fd_train() if db_type is self.DBType.Train \
            else self.fd_val()
        cur_loss, cur_pred, self.cur_inputs = \
            sess.run( [self.cost, self.pred, self.inputs], self.fd)
        cur_dist = self.compute_dist(cur_pred, self.cur_inputs[1])
        return cur_loss, cur_dist


    def train(self):
        self.create_network()
        self.create_optimizer()
        self.create_saver()
        training_iters = self.conf.dl_steps
        num_val_rep = self.conf.numTest / self.conf.batch_size + 1
        learning_rate = 0.0001

        with tf.Session() as sess:
            sess.run(tf.variables_initializer(self.get_var_list()))
            self.init_td()

            start = time.time()
            for step in range(0, training_iters + 1):
                self.train_step(step, sess, learning_rate, training_iters)
                if step % self.conf.display_step == 0:
                    end = time.time()
                    print('Time required to train: {}'.format(end-start))
                    train_loss, train_dist = self.compute_train_data(sess, self.DBType.Train)
                    val_loss = 0.
                    val_dist = 0.
                    for _ in range(num_val_rep):
                       cur_loss, cur_dist = self.compute_train_data(sess, self.DBType.Val)
                       val_loss += cur_loss
                       val_dist += cur_dist
                    val_loss = val_loss / num_val_rep
                    val_dist = val_dist / num_val_rep
                    cur_dict = {'step': step,
                               'train_loss': train_loss, 'val_loss': val_loss,
                               'train_dist': train_dist, 'val_dist': val_dist}
                    self.update_td(cur_dict)
                    start = end
                if step % self.conf.save_step == 0:
                    self.save(sess, step)
                if step % self.conf.save_td_step == 0:
                    self.save_td()
            print("Optimization Finished!")
            self.save(sess, training_iters)
            self.save_td()
        tf.reset_default_graph()

