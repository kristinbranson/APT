import pickle
import numpy as np
from matplotlib import pyplot as plt

name = 'stephen'
in_file = '/home/mayank/Dropbox (HHMI)/PoseEstimation/Results/stephen_gt_numbers.p'
db =

with open(in_file,'r') as f:
    D = pickle.load(f)

##
P = {}
all_t = {}
for net in D.keys():
    cur_out = D[net]
    V = []
    n_out = min([len(x) for x in cur_out])
    t = np.zeros([len(cur_out),n_out])
    for view in range(len(cur_out)):
        K = []
        for out in range(n_out):
            preds = cur_out[view][out][0]
            labels = cur_out[view][out][1]
            dd = np.sqrt(np.sum((preds-labels)**2,axis=-1))
            pp = np.percentile(dd,[50,75,90,95,98,99],axis=0)
            K.append(pp)
            #t[view,out] = cur_out[view][out][7]
        V.append(K)
    P[net] = np.array(V)

    if net == 'unet':
        t = np.tile( np.arange(n_out)*20 ,[2,1])
    elif net == 'openpose':
        t = np.tile(np.arange(n_out) * 19, [2, 1])
    elif net == 'deeplabcut':
        t = np.tile( np.arange(n_out)*35 ,[2,1])
    elif net == 'leap':
        t = np.tile(np.arange(n_out) * 3, [2, 1])

    all_t[net] = t



##

f,ax = plt.subplots(2,3,sharex=True,sharey=True)
ax = ax.flatten()
for z in range(P['unet'].shape[-1]):
    for net in P.keys():
        ax[z].plot(all_t[net][0,:],P[net][0,:,z,:],label=net)
plt.legend()

##

import multiResData
import PoseTools
f, ax = plt.subplots(2,4)
for view in range(2):
    if view == 1:
        from stephenHeadConfig import  conf as conf
        A = multiResData.read_and_decode_without_session('/home/mayank/work/poseTF/cacheHead/train_TF.tfrecords', conf,
                                                         range(10))
    else:
        from stephenHeadConfig import  sideconf as conf
        A = multiResData.read_and_decode_without_session('/home/mayank/work/poseTF/cacheHeadSide/train_TF.tfrecords', conf,range(10))


    for ndx, net in enumerate(P.keys()):
        if net == 'deeplabcut':
            sndx = 5
        elif net == 'openpose':
            sndx = 10
        else:
            sndx = -1
        PoseTools.create_result_image(A[0][0][:,:,0],A[1][0],P[net][view,sndx,:,:],ax[view,ndx])
        if view == 0:
            ax[view,ndx].set_title(net)
