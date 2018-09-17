import train
train.train()

##  Results for stephen.

import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.interactive(True)
import matplotlib.pyplot as plt

sys.path.append('/home/mayank/work/poseTF')

K = pd.read_hdf('Results/DeepCut_resnetresnet_50_100shuffle1_250000forTask:stephen-val.h5')
L = pd.read_hdf('/home/mayank/work/deepcut/Generating_a_Training_Set/data-stephen-val/CollectedData_stephen.h5')

K_arr = []
for k in K.keys():
    K_arr.append(K[k].tolist())
del K_arr[2::3]
K_arr = np.array(K_arr).T
L_arr = []
for k in L.keys():
    L_arr.append(L[k].tolist())
L_arr = np.array(L_arr).T

import  imageio
import  PoseTools

im = imageio.imread('/home/mayank/work/deepcut/Generating_a_Training_Set/data-stephen-val/stephenvideo1/img_00032.png')
dd = np.sqrt((K_arr[:,::2]-L_arr[:,::2])**2 +(K_arr[:,1::2]-L_arr[:,1::2])**2)
pp = np.percentile(dd,[90,95,98,99,99.5],axis=0)

locs = np.reshape(L_arr[32,:],[5,2])
PoseTools.create_result_image(im,locs,pp)


## Results for alice

import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.interactive(True)
import matplotlib.pyplot as plt

sys.path.append('/home/mayank/work/poseTF')

K = pd.read_hdf('Results/DeepCut_resnetresnet_50_100shuffle1_500000forTask:alice-val.h5')
L = pd.read_hdf('/home/mayank/work/deepcut/Generating_a_Training_Set/data-alice-val/CollectedData_alice.h5')

K_arr = []
for k in K.keys():
    K_arr.append(K[k].tolist())
del K_arr[2::3]
K_arr = np.array(K_arr).T
L_arr = []
for k in L.keys():
    L_arr.append(L[k].tolist())
L_arr = np.array(L_arr).T


import  imageio
import  PoseTools

im = imageio.imread('/home/mayank/work/deepcut/Generating_a_Training_Set/data-alice-val/alicevideo1/img_00032.png')
dd = np.sqrt((K_arr[:,::2]-L_arr[:,::2])**2 +(K_arr[:,1::2]-L_arr[:,1::2])**2)
pp = np.percentile(dd,[90,95,98,99,99.5],axis=0)

locs = np.reshape(L_arr[32,:],[17,2])
PoseTools.create_result_image(im,locs,pp)
