
# coding: utf-8

# In[1]:

# all the f'ing imports
import scipy.io as sio
import os,sys
import myutils
import re
from stephenBodyConfig import conf as conf
import shutil

get_ipython().magic(u'matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
import math
import lmdb
import caffe
from random import randint,sample
import pickle
import h5py


import mpld3
mpld3.enable_notebook()

import multiResData
reload(multiResData)

multiResData.createDB(conf)


# In[2]:

# all the f'ing imports
import scipy.io as sio
import os,sys
import myutils
import re
from stephenBodyConfig import sideconf as conf
import shutil

get_ipython().magic(u'matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import cv2
import math
import lmdb
import caffe
from random import randint,sample
import pickle
import h5py


import mpld3
mpld3.enable_notebook()

import multiResData
reload(multiResData)

multiResData.createDB(conf)


# In[1]:

import PoseTrain
import tensorflow as tf

tf.reset_default_graph()
from stephenBodyConfig import conf as conf
pobj = PoseTrain.PoseTrain(conf)
pobj.baseTrain(restore=False)

tf.reset_default_graph()
from stephenBodyConfig import sideconf as conf
pobj = PoseTrain.PoseTrain(conf)
pobj.baseTrain(restore=False)


# In[1]:

import localSetup
import PoseTools
import multiResData
import os
import re
import tensorflow as tf
from scipy import io
import cv2
from cvc import cvc

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

redo = True
makemovie = False

for view in range(2):
    tf.reset_default_graph() 
    if view == 1:
        # For SIDE

        from stephenBodyConfig import sideconf as conf
        conf.useMRF = False
        outtype = 1
        extrastr = '_side'
    else:
#         For FRONT
        from stephenBodyConfig import conf as conf
        conf.useMRF = False
        outtype = 1
        extrastr = '_front'

    # conf.batch_size = 1

    self = PoseTools.create_network(conf, outtype)
    sess = tf.InteractiveSession()
    PoseTools.init_network(self, sess, outtype)

    scale = conf.rescale*conf.pool_scale
    _,valmovies = multiResData.get_movie_lists(conf)
    for ndx in range(len(valmovies)):
        mname,_ = os.path.splitext(os.path.basename(valmovies[ndx]))00
        oname = re.sub('!','__',conf.getexpname(valmovies[ndx]))
        pname = os.path.join(localSetup.bdir, 'data', 'out', oname + extrastr)
        if os.path.isfile(pname + '.mat') and not redo:
            continue


        if not os.path.isfile(valmovies[ndx]):
            continue

        predList = PoseTools.classify_movie(conf, valmovies[ndx], outtype, self, sess)
    #     PoseTools.createPredMovie(conf,predList,valmovies[ndx],pname + '.avi',outtype)


        cap = cv2.VideoCapture(valmovies[ndx])
        height = int(cap.get(cvc.FRAME_HEIGHT))
        width = int(cap.get(cvc.FRAME_WIDTH))
        orig_crop_loc = conf.cropLoc[(height,width)]
        crop_loc = [x/scale for x in orig_crop_loc] 
        end_pad = [height/scale-crop_loc[0]-conf.imsz[0]/scale,width/scale-crop_loc[1]-conf.imsz[1]/scale]
        pp = [(0,0),(crop_loc[0],end_pad[0]),(crop_loc[1],end_pad[1]),(0,0),(0,0)]
        predScores = np.pad(predList[1],pp,mode='constant',constant_values=-1.)

        predLocs = predList[0]
        predLocs[:,:,:,0] += orig_crop_loc[1]
        predLocs[:,:,:,1] += orig_crop_loc[0]

        io.savemat(pname + '.mat',{'locs':predLocs,'scores':predScores[...,0],'expname':valmovies[ndx]})
        print 'Done:%s'%oname


