
## Setup

import APT_interface as apt
import RNN_postprocess
import tensorflow as tf
import os
import easydict
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.reset_default_graph()

exp_name = 'postprocess'
view = 0
mdn_name = 'deepnet'
lbl_file = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4992_gtcomplete_cacheddata_updatedAndPpdbManuallyCopied20190402_dlstripped.lbl'

conf = apt.create_conf(lbl_file,view,exp_name,'/nrs/branson/mayank/apt_cache','mdn')
conf.n_steps = 2

conf.rrange = 30
conf.trange = 10
conf.mdn_use_unet_loss = True
conf.dl_steps = 40000
conf.decay_steps = 20000
conf.save_step = 5000
conf.batch_size = 8
conf.normalize_img_mean = False
conf.maxckpt = 20


## Train MDN
args = easydict.EasyDict
args.skip_db = False
args.use_cache = True
args.train_name = mdn_name
apt.train_mdn(conf,args,False,True)


##

split_file = os.path.join(conf.cachedir,'splitdata.json')
self = RNN_postprocess.RNN_pp(conf,mdn_name,
                              name = 'rnn_pp',
                              data_name='rnn_pp_' + mdn_name)
self.rnn_pp_hist = 32
self.train_rep = 5
self.create_db(split_file = split_file)
tf.reset_default_graph()
self.net_type = 'conv'
# self.train()
tf.reset_default_graph()
V = self.classify_val()
##

