from __future__ import division
from __future__ import print_function

# coding: utf-8

# In[2]:

from builtins import range
from past.utils import old_div
import scipy.io as sio
import os,sys
sys.path.append('/home/mayank/work/pyutils')
import myutils
import re
import pawconfig as conf
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


# In[ ]:

def findLocalDirs():
    L = sio.loadmat(conf.labelfile)

    seldirs = [False]*len(L['expdirs'][0])
    localdirs =[]
    for ndx,dirname in enumerate(L['expdirs'][0]):
        curdir = dirname[0]
        expname = os.path.basename(curdir)
        splits = expname.split('_')
        localdirname = os.path.join(conf.viddir,'_'.join(splits[0:2]),expname)
        localdirs.append(localdirname)
        if os.path.isdir(localdirname) & (re.search(conf.ptn,expname) is not None):
            seldirs[ndx] = True
    return localdirs,seldirs


# In[4]:

def createValdata(force):
    
    outfile = os.path.join(conf.cachedir,conf.valdatafilename)
    if ~force & os.path.isfile(outfile):
        return

    L = sio.loadmat(conf.labelfile)
    nexps = len(L['expdirs'][0])
    print(nexps)
    isval = sample(list(range(nexps)),int(nexps*conf.valratio))
    localdirs,seldirs = findLocalDirs()
    with open(outfile,'w') as f:
        pickle.dump([isval,localdirs,seldirs],f)


# In[6]:

def loadValdata():
    
    outfile = os.path.join(conf.cachedir,conf.valdatafilename)
    assert os.path.isfile(outfile),"valdatafile doesn't exist"

    with open(outfile,'r') as f:
        isval,localdirs,seldirs = pickle.load(f)
    return isval,localdirs,seldirs


# In[ ]:

def createDatum(curp,label):
    datum = caffe.proto.caffe_pb2.Datum()
    datum.channels = curp.shape[0]
    datum.height = curp.shape[1]
    datum.width = curp.shape[2]
    datum.data = curp.tostring()  # or .tobytes() if numpy >= 1.9
    datum.label = label
    return datum


# In[ ]:

def getpatch(cap,fnum,curloc):
    # matlab sometimes can access an additional frame at the end
    # which others can't.
    curp = None
    psz = conf.sel_sz

    if fnum > cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
        if fnum > cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)+1:
            raise ValueError('Accessing frames beyond the length of the video')
        return curp
    framein = myutils.readframe(cap,fnum-1)
    framein = framein[:,0:(old_div(framein.shape[1],2)),:]

    testp = myutils.padgrab(framein,0,curloc[1]-old_div(psz,2),curloc[1]+old_div(psz,2),
                           curloc[0]-old_div(psz,2),curloc[0]+old_div(psz,2),0,framein.shape[2])
    curp = np.array(scalepatches(testp,conf.scale,conf.numscale,conf.rescale))
    return curp


# In[ ]:

def createID(expname,curloc,fnum):
    str_id = '{:08d}:{}:x{:d}:y{:d}:t{:d}'.format(randint(0,1e8),
           expname,curloc[0],curloc[1],fnum)
    return str_id


# In[ ]:

def decodeID(keystr):
    vv = re.findall('(\d+):(.*):x(\d+):y(\d+):t(\d+)',keystr)[0]
    return vv[1],(vv[2],vv[3]),vv[4]


# In[ ]:

def closeToPos(curloc,posloc):
    d2pos = math.sqrt( (curloc[0]-posloc[0])**2 + (curloc[1]-posloc[1])**2)
    return d2pos < conf.dist2pos


# In[ ]:

def createDB():

    L = sio.loadmat(conf.labelfile)
    pts = L['pts']
    ts = L['ts']
    expid = L['expidx']
    
    count = 0; valcount = 0
    
    psz = conf.sel_sz
    map_size = 100000*conf.psz**2*3
    
    createValdata(False)
    isval,localdirs,seldirs = loadValdata()
    
    lmdbfilename =os.path.join(conf.cachedir,conf.trainfilename)
    vallmdbfilename =os.path.join(conf.cachedir,conf.valfilename)
    if os.path.isdir(lmdbfilename):
        shutil.rmtree(lmdbfilename)
    if os.path.isdir(vallmdbfilename):
        shutil.rmtree(vallmdbfilename)
    
    env = lmdb.open(lmdbfilename, map_size=map_size)
    valenv = lmdb.open(vallmdbfilename, map_size=map_size)

    
    with env.begin(write=True) as txn,valenv.begin(write=True) as valtxn:

        for ndx,dirname in enumerate(localdirs):
            if not seldirs[ndx]:
                continue

            expname = os.path.basename(dirname)
            frames = np.where(expid[0,:] == (ndx + 1))[0]
            curdir = localdirs[ndx]
            cap = cv2.VideoCapture(os.path.join(curdir,'movie_comb.avi'))
            
            curtxn = valtxn if isval.count(ndx) else txn
                
            for curl in frames:

                fnum = ts[0,curl]
                curloc = np.round(pts[0,:,curl]).astype('int')
                if fnum > cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
                    if fnum > cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)+1:
                        raise ValueError('Accessing frames beyond the length of the video')
                    continue
                
                framein = myutils.readframe(cap,fnum-1)
                framein = framein[:,0:(old_div(framein.shape[1],2)),0:1]

                datum = createDatum(framein,1)
                str_id = createID(expname,curloc,fnum)
                curtxn.put(str_id.encode('ascii'), datum.SerializeToString())

                if isval.count(ndx):
                    valcount+=1
                else:
                    count+=1
                    
            cap.release() # close the movie handles
            print('Done %d of %d movies' % (ndx,len(localdirs)))
    env.close() # close the database
    valenv.close()
    print('%d,%d number of pos examples added to the db and valdb' %(count,valcount))
    
    


# In[1]:

def createPos():
    L = sio.loadmat(conf.labelfile)
    pts = L['pts']
    ts = L['ts']
    expid = L['expidx']
    
    count = 0; valcount = 0
    
    psz = conf.sel_sz
    map_size = 100000*conf.psz**2*3
    
    createValdata(False)
    isval,localdirs,seldirs = loadValdata()
    
    lmdbfilename =os.path.join(conf.cachedir,conf.trainfilename)
    vallmdbfilename =os.path.join(conf.cachedir,conf.valfilename)
    if os.path.isdir(lmdbfilename):
        shutil.rmtree(lmdbfilename)
    if os.path.isdir(vallmdbfilename):
        shutil.rmtree(vallmdbfilename)
    
    env = lmdb.open(lmdbfilename, map_size=map_size)
    valenv = lmdb.open(vallmdbfilename, map_size=map_size)
    
    with env.begin(write=True) as txn,valenv.begin(write=True) as valtxn:

        for ndx,dirname in enumerate(localdirs):
            if not seldirs[ndx]:
                continue

            expname = os.path.basename(dirname)
            frames = np.where(expid[0,:] == (ndx + 1))[0]
            curdir = localdirs[ndx]
            cap = cv2.VideoCapture(os.path.join(curdir,'movie_comb.avi'))
            
            curtxn = valtxn if isval.count(ndx) else txn
                
            for curl in frames:

                fnum = ts[0,curl]
                curloc = np.round(pts[0,:,curl]).astype('int')

                curp = getpatch(cap,fnum,curloc)
                if curp is None:
                    continue
                datum = createDatum(curp,1)
                str_id = createID(expname,curloc,1,fnum)
                curtxn.put(str_id.encode('ascii'), datum.SerializeToString())

                if isval.count(ndx):
                    valcount+=1
                else:
                    count+=1
                    
            cap.release() # close the movie handles
            print('Done %d of %d movies' % (ndx,len(localdirs)))
    env.close() # close the database
    valenv.close()
    print('%d,%d number of pos examples added to the db and valdb' %(count,valcount))
    
    


# In[1]:

def addRandomNeg():
    L = sio.loadmat(conf.labelfile)
    pts = L['pts']
    ts = L['ts']
    expid = L['expidx']
    
    count = 0; valcount = 0
    
    psz = conf.sel_sz
    map_size = 100000*conf.psz**2*3
    
    isval,localdirs,seldirs = loadValdata()
    
    lmdbfilename =os.path.join(conf.cachedir,conf.trainfilename)
    vallmdbfilename =os.path.join(conf.cachedir,conf.valfilename)
    env = lmdb.open(lmdbfilename, map_size=map_size)
    valenv = lmdb.open(vallmdbfilename, map_size=map_size)
    
    with env.begin(write=True) as txn,valenv.begin(write=True) as valtxn:

        for ndx,dirname in enumerate(localdirs):
            if not seldirs[ndx]:
                continue

            expname = os.path.basename(dirname)
            frames = np.where(expid[0,:] == (ndx + 1))[0]
            curdir = localdirs[ndx]
            cap = cv2.VideoCapture(os.path.join(curdir,'movie_comb.avi'))
            
            curtxn = valtxn if isval.count(ndx) else txn
                
            for curl in frames:

                fnum = ts[0,curl]
                width = old_div(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH),2)
                height = old_div(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT),2)
                posloc = pts[0,:,curl]

                trycount = 0
                for curneg in range(conf.numNegPerPos):
                    
                    curloc =[randint(0,width),randint(0,height)]
                    while closeToPos(curloc,posloc) and trycount<30:
                        curloc =[randint(0,width),randint(0,height)]
                        trycount = trycount+1

                    if closeToPos(curloc,posloc):
                        break
                        
                    curp = getpatch(cap,fnum,curloc)
                    if curp is None:
                        continue

                    datum = createDatum(curp,0)
                    str_id = createID(expname,curloc,0,fnum)
                    curtxn.put(str_id.encode('ascii'), datum.SerializeToString())

                    if isval.count(ndx):
                        valcount+=1
                    else:
                        count+=1

            cap.release() # close the movie handles
            print('Done %d of %d movies' % (ndx,len(localdirs)))
    env.close() # close the database
    valenv.close()
    print('%d,%d number of pos examples added to the db and valdb' %(count,valcount))
    
    

