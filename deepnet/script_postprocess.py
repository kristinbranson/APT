
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
# apt.train_mdn(conf,args,False,True)


##

split_file = os.path.join(conf.cachedir,'splitdata.json')
self = RNN_postprocess.RNN_pp(conf,mdn_name,
                              name = 'rnn_pp',
                              data_name='rnn_pp_' + mdn_name)
self.rnn_pp_hist = 32
self.train_rep = 5
# self.create_db(split_file = split_file)
tf.reset_default_graph()
self.net_type = 'simple_fc'
self.train()
tf.reset_default_graph()
V = self.classify_val()
##


data_file = os.path.join(self.conf.cachedir,self.data_name + '.p')
import pickle
with open(data_file,'r') as f:
    X = pickle.load(f)

t_labels = np.array([x[1] for x in X[0]]).reshape([-1,self.conf.n_classes*2])
t_inputs_1 = np.array([x[0] for x in X[0]]).reshape([-1,2*self.rnn_pp_hist,self.conf.n_classes*2])
t_inputs_2 = np.array([x[3] for x in X[0]]).reshape([-1,2*self.rnn_pp_hist,self.conf.n_classes*2])
t_inputs = np.concatenate([t_inputs_1, t_inputs_2],axis=-1)
v_labels = np.array([x[1] for x in X[1]]).reshape([-1,self.conf.n_classes*2])
v_inputs_1 = np.array([x[0] for x in X[1]]).reshape([-1,2*self.rnn_pp_hist,self.conf.n_classes*2])
v_inputs_2 = np.array([x[3] for x in X[1]]).reshape([-1,2*self.rnn_pp_hist,self.conf.n_classes*2])

import multiResData
tt = 'val'
if tt == 'val':
    info_t = np.array([x[2] for x in X[1]])
    H = multiResData.read_and_decode_without_session(os.path.join(conf.cachedir,'val_TF.tfrecords'),conf,())
    t1 = v_inputs_1.reshape([-1,2*rx,self.conf.n_classes,2])
    t2 = v_labels.reshape([-1,self.conf.n_classes,2])

else:
    info_t = np.array([x[2] for x in X[0]])
    H = multiResData.read_and_decode_without_session(os.path.join(conf.cachedir,'train_TF.tfrecords'),conf,())
    t1 = t_inputs_1.reshape([-1,2*rx,self.conf.n_classes,2])
    t2 = t_labels.reshape([-1,self.conf.n_classes,2])

ims = np.array(H[0])
locs = np.array(H[1])
info = np.array(H[2])


##
import PoseTools
cm = PoseTools.get_cmap(5)
rx = self.rnn_pp_hist
dd = np.sqrt(np.sum((t1[:,rx,:,:]-t2)**2,axis=-1))

far_ndx = np.where(np.any(dd>15,axis=-1))[0]
ndx = np.random.choice(far_ndx)

selndx = np.where( np.all( (info_t[ndx,:] - info)==0,1))[0][0]
f = plt.figure(figsize=(18,18))
plt.imshow(ims[selndx,:,:,0],'gray')
for pt in range(conf.n_classes):
    plt.plot(t1[ndx,rx-8:rx+8,pt,0],t1[ndx,rx-8:rx+8,pt,1],c=cm[pt,:])
    plt.scatter(t2[ndx,pt,0],t2[ndx,pt,1],c=cm[pt,:])
    plt.plot([t1[ndx,rx,pt,0],t2[ndx,pt,0]],[t1[ndx,rx,pt,1],t2[ndx,pt,1]],c=cm[pt,:],lw=5)
print dd[ndx,:]
##

jj = []
ws = [0,1,2,3,5,10]
f,ax = plt.subplots(2,3)
ax = ax.flatten()
dd = np.sqrt(np.sum((t1[:,rx,:,:]-t2)**2,axis=-1))
far_ndx = np.where(np.any(dd>15,axis=-1))[0]
for ndx,w in enumerate(ws):
    pp = t1[far_ndx,rx-w:rx+w+1,:,:]
    ll = t2[far_ndx,np.newaxis,:,:]
    dd = np.sqrt(np.sum((pp-ll)**2,axis=-1))
    rr = np.median(dd,axis=1)
    ss = np.percentile(rr,[50,75,90,95],axis=0)
    # ax[ndx].imshow(ims[0,:,:,0],'gray')
    jj.append(ss)

##

