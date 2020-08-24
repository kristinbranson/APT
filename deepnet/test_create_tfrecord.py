import numpy as np
import APT_interface as apt
import multiResData
import os
import pdb
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import cv2
from cvc import cvc
import pickle

def write_tfrecords(lbl_file,view,cache_dir,net,nsamples):

    cache_conf = apt.create_conf(lbl_file,view,'cache',cache_dir,net_type=net)
    nocache_conf = apt.create_conf(lbl_file,view,'nocache',cache_dir,net_type=net)
    nocache_file = os.path.join(cache_dir,'test_nocache.tfrecords')
    cache_file = os.path.join(cache_dir,'test_cache.tfrecords')
    print('writing using cache to %s'%cache_file)
    apt.create_tfrecord(cache_conf,split=False,split_file=None,use_cache=True,on_gt=False,db_files=[cache_file],max_nsamples=nsamples)
    print('writing not using cache to %s'%nocache_file)
    apt.create_tfrecord(nocache_conf,split=False,split_file=None,use_cache=False,on_gt=False,db_files=[nocache_file],max_nsamples=nsamples)

def read_tfrecords(lbl_file,view,cache_dir,net,nsamples):

    pti = 5
    nocache_conf = apt.create_conf(lbl_file,view,'nocache',cache_dir,net_type='mdn')
    cache_conf = apt.create_conf(lbl_file,view,'cache',cache_dir,net_type='mdn')
    nocache_proj_name = nocache_conf.expname
    cache_proj_name = cache_conf.expname
    nocache_db_file = os.path.join(cache_dir,nocache_proj_name,net,'view_%d'%view,'nocache','train_TF.tfrecords')
    cache_db_file = os.path.join(cache_dir,cache_proj_name,net,'view_%d'%view,'cache','train_TF.tfrecords')
    nocache_tf_iterator = multiResData.tf_reader(nocache_conf, nocache_db_file, False)
    nocache_tf_iterator.batch_size = 1
    nocache_read_fn = nocache_tf_iterator.next

    cache_tf_iterator = multiResData.tf_reader(cache_conf, cache_db_file, False)
    cache_tf_iterator.batch_size = 1
    cache_read_fn = cache_tf_iterator.next

    fig,ax = plt.subplots(3,nsamples,sharex='all',sharey='all')

    for samplei in range(nsamples):
    
        nocache_db = nocache_read_fn()
        nocache_im = nocache_db[0]
        imsz = (nocache_im.shape[1],nocache_im.shape[2])
        nocache_im = nocache_im.reshape(imsz)
        nocache_loc = nocache_db[1]

        cache_db = cache_read_fn()
        cache_im = cache_db[0]
        cache_im = cache_im.reshape(imsz)
        cache_loc = cache_db[1]

        diff_im = cache_im.astype(float) - nocache_im.astype(float)

        axcurr = ax[0,samplei]
        axcurr.imshow(nocache_im,vmin=0,vmax=255)
        axcurr.plot(nocache_loc[0,pti,0],nocache_loc[0,pti,1],'rx')
        axcurr.set_title('%d nocache'%samplei)

        axcurr = ax[1,samplei]
        axcurr.imshow(cache_im,vmin=0,vmax=255)
        axcurr.plot(cache_loc[0,pti,0],cache_loc[0,pti,1],'m+')
        axcurr.set_title('%d cache'%samplei)
        
        axcurr = ax[2,samplei]
        axcurr.imshow(diff_im,vmin=-3,vmax=3)
        axcurr.plot(nocache_loc[0,pti,0],nocache_loc[0,pti,1],'rx')
        axcurr.plot(cache_loc[0,pti,0],cache_loc[0,pti,1],'m+')
        axcurr.set_title('%d cache - nocache'%samplei)


    plt.show()
    
    #pdb.set_trace()

def test_cropping(lbl_file,view,cache_dir,net):

    nocache_conf = apt.create_conf(lbl_file,view,'nocache',cache_dir,net_type='mdn')
    cache_conf = apt.create_conf(lbl_file,view,'cache',cache_dir,net_type='mdn')
    nocache_proj_name = nocache_conf.expname
    cache_proj_name = cache_conf.expname
    nocache_db_file = os.path.join(cache_dir,nocache_proj_name,net,'view_%d'%view,'nocache','train_TF.tfrecords')
    cache_db_file = os.path.join(cache_dir,cache_proj_name,net,'view_%d'%view,'cache','train_TF.tfrecords')
    nocache_tf_iterator = multiResData.tf_reader(nocache_conf, nocache_db_file, False)
    nocache_tf_iterator.batch_size = 1
    nocache_read_fn = nocache_tf_iterator.next

    cache_tf_iterator = multiResData.tf_reader(cache_conf, cache_db_file, False)
    cache_tf_iterator.batch_size = 1
    cache_read_fn = cache_tf_iterator.next

    nocache_db = nocache_read_fn()
    nocache_im = nocache_db[0]
    imsz = (nocache_im.shape[1],nocache_im.shape[2])
    nocache_im = nocache_im.reshape(imsz)
    nocache_loc = nocache_db[1]
    
    cache_db = cache_read_fn()
    cache_im = cache_db[0]
    cache_im = cache_im.reshape(imsz)
    cache_loc = cache_db[1]
    
    fn = 'test_multiResData.pkl'
    data = pickle.load(open(fn,'rb'))

    psz_x = 181
    psz_y = 181
    im = data['im']
    x = data['x']
    y = data['y']
    theta = data['theta']
    T = np.array([[1, 0, 0], [0, 1, 0], [-x + float(psz_x) / 2 - 0.5, -y + float(psz_y) / 2 - 0.5, 1]]).astype('float')
    R1 = cv2.getRotationMatrix2D((float(psz_x) / 2 - 0.5, float(psz_y) / 2 - 0.5), theta * 180 / np.pi, 1)
    R = np.eye(3)
    R[:, :2] = R1.T
    A_full = np.matmul(T,R)

    A = A_full[:,:2].T
    rpatch = cv2.warpAffine(im, A, (psz_x,psz_y),flags=cv2.INTER_LINEAR)

    cache_diff_im = cache_im.astype(float) - rpatch.astype(float)
    nocache_diff_im = nocache_im.astype(float) - rpatch.astype(float)
    diff_im = cache_im.astype(float) - nocache_im.astype(float)
    
    fig,ax = plt.subplots(1,5,sharex='all',sharey='all')
    axcurr = ax[0]
    axcurr.imshow(rpatch,vmin=0,vmax=255)
    axcurr.set_title('computed')
    axcurr = ax[1]
    axcurr.imshow(cache_im,vmin=0,vmax=255)
    axcurr.set_title('cache')
    axcurr = ax[2]
    axcurr.imshow(cache_diff_im,vmin=-10,vmax=10)
    axcurr.set_title('cache - computed')
    axcurr = ax[3]
    axcurr.imshow(nocache_diff_im,vmin=-10,vmax=10)
    axcurr.set_title('nocache - computed')
    axcurr = ax[4]
    axcurr.imshow(diff_im,vmin=-10,vmax=10)
    axcurr.set_title('cache - nocache')
    plt.show()
    
if __name__ == '__main__':
    lbl_file = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20200317_stripped20200403_new_skl_20200817.lbl'
    nsamples = 10
    view = 0
    cache_dir = '/nrs/branson/apt_cache'
    net = 'mdn'

    #test_cropping(lbl_file,view,cache_dir,net)
    #write_tfrecords(lbl_file,view,cache_dir,net,nsamples)
    read_tfrecords(lbl_file,view,cache_dir,net,nsamples)
    
