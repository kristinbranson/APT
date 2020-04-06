# script for testing python video reader to matlab video reader on windows.
import cv2
import scipy.io as sio
import  numpy as np
import movies
import hdf5storage
import os
import glob
import imageio
import logging

on_compressed = True
dd_all = []
for machine in ['GLNXA64_h06u08.int.janelia.org_R2018a',
                'PCWIN64_User7910_R2017b',
                'GLNXA64_verman-ws1.hhmi.org_R2018a']:
    bdir = '/groups/branson/bransonlab/apt/videotest/'
    mov = 'C001H001S0001'
    if on_compressed:
        mov = mov + '_c'

    frame_dir = bdir + 'VideoTest_' + mov + '_'+ machine
    mov = bdir + mov + '.avi'

    oo = glob.glob(frame_dir + '*')
    frame_dir = oo[0]

    log = logging.getLogger()
    log.setLevel(logging.INFO)

    cap = movies.Movie(mov)
    nfr = 1000 # cap.get_n_frames()
    dd = []
    for ndx in range(20):
        curfr = np.random.randint(nfr)
        curim = cap.get_frame(curfr)[0][:, :, 0]
        mat_im = imageio.imread(os.path.join(frame_dir,'sr_{:06d}.png'.format(curfr+1)))
        if np.ndim(mat_im)>2:
            mat_im = mat_im[...,0]
        ff = curim.astype('float64')- mat_im.astype('float64')
        cur_i = [curfr, np.abs(ff[2:-1,2:-1]).max()]
        dd.append(cur_i)
    dd_all.append(dd)
print dd_all


##
if False:
    mov = '/groups/branson/bransonlab/al/C001H001S0007_c.avi'
    win_mat = '/groups/branson/bransonlab/al/C001H001S0007_c.avi.readFrames_1450_allenwin_16b.mat'

    py_mat = '/groups/branson/home/kabram/temp/C001H001S0007_c.avi.readFrames_1450_kabram_py.mat'
    H = sio.loadmat(win_mat)

    nfr = H['I'].shape[0]
    ims = np.zeros([512,768,nfr])
    for ndx in range(nfr):
        ims[:,:,ndx] = H['I'][ndx][0]

    import os
    if os.path.exists(py_mat):
        P = hdf5storage.loadmat(py_mat)
        py_ims = P['I']
    else:
        py_ims = None
    ## random access

    cap = movies.Movie(mov)
    dd = []
    for ndx in range(10):
        curfr = np.random.randint(nfr)
        curim = cap.get_frame(curfr)[0][:,:,0]
        ff = curim.astype('float64')+1-ims[:,:,curfr]
        cur_i = [curfr, np.abs(ff).max()]
        if py_ims is not Noneimp:
            ffp = curim.astype('float64')-py_ims[:,:,curfr]
            cur_i.append(np.abs(ffp).max())
        # dd.append([curfr, int(np.percentile(np.abs(ff),100-0.002))])
        dd.append(cur_i)

    print dd

    cap.close()

    ##

    if not os.path.exists(py_mat):
        cap = movies.Movie(mov)
        dd = []
        py_ims = np.zeros([512, 768, nfr])
        for ndx in range(nfr):
            curfr = ndx
            curim = cap.get_frame(curfr)[0][:, :, 0]
            py_ims[:, :, ndx] = curim

        hdf5storage.savemat(py_mat,{'I':py_ims})

