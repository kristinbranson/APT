from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
import tensorflow as tf
# from multiResData import *
import h5py
import pickle
import sys
import os
import PoseTrain
import multiResData
import numpy as np
import argparse
import importlib
import re
from matplotlib import pyplot as plt
import PoseTools
from scipy import io
import cv2
from cvc import cvc
import copy
import math
##
def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("-fold",dest="fold",
                      help="fold number",
                      required=True)
    parser.add_argument("-gpu",dest='gpunum',
                        help="GPU to use",required=True)
    parser.add_argument("-bs",dest='batch_size',type=int,
                        help="batch_size to use [optional]")
    parser.add_argument("-onlydb",dest="onlydb",
                      help="if specified will only create dbs",
                      action="store_true")
    parser.add_argument("-conf",dest="configfile",
                      help="configfile to use (without the .py ext)",
                      required=True)
    parser.add_argument("-train",dest='train',action="store_true",
                        help="Train")
    parser.add_argument("-detect",dest='detect',action="store_true",
                        help="Detect")
    parser.add_argument("-fine_train",dest='fine_train',action="store_true",
                        help="Fine Train")
    parser.add_argument("-fine_detect",dest='fine_detect',action="store_true",
                        help="Fine Detection")
    parser.add_argument("-r",dest="redo",
                      help="if specified will recompute everything",
                      action="store_true")
    parser.add_argument("-o",dest="outdir",
                      help="temporary output directory to store intermediate computations")
    parser.add_argument("-cname",dest="confname",
                      help="config name")
    parser.add_argument("-outtype",dest="outtype",
                      help="out type: 2 for mrf, 1 for pd",type=int)
    parser.add_argument("-extra_str",dest="extra_str",
                      help="extra str to attach to output files")
    parser.add_argument("-results_dir",dest="results_dir",
                      help="Directory with tracking results")

    args = parser.parse_args(argv[1:])

    if args.train is None and args.detect is None:
        args.train = True
        args.detect = True


    curfold = args.fold
    curgpu = args.gpunum
    if args.batch_size is None:
        batch_size = -1
    else:
        batch_size = args.batch_size

    if args.onlydb is None:
        onlydb = False
    else:
        onlydb = args.onlydb

    if args.redo is None:
        args.redo = False

    if args.confname is None:
        args.confname = 'conf'
    if args.outtype is None:
        args.outtype = 2

    if args.train:

        trainfold(conffile=args.configfile,curfold=curfold,
                  curgpu=curgpu,batch_size=batch_size,onlydb=onlydb,
                  confname=args.confname)

    if args.detect:
        classifyfold(conffile=args.configfile,curfold=curfold,
                     curgpu=curgpu,batch_size=batch_size,redo=args.redo,
                     outdir=args.outdir,confname=args.confname,
                     outtype=args.outtype,extra_str=args.extra_str)


    if args.fine_train:
        trainfold_fine(conffile=args.configfile, curfold=curfold,
                  curgpu=curgpu, batch_size=batch_size,
                  confname=args.confname)

    if args.fine_detect:
        if args.results_dir is None:
            print('Need results dir for fine detection')
            return
        classifyfold_fine(conffile=args.configfile,curfold=curfold,
                     curgpu=curgpu,batch_size=batch_size,redo=args.redo,
                     outdir=args.outdir,results_dir=args.results_dir,confname=args.confname,
                     extra_str=args.extra_str)



def createvaldataJan(conf):
    from janLegConfig import conf
    ##
    L  = h5py.File(conf.labelfile)
    localdirs,seldirs = multiResData.find_local_dirs(conf)

    ##
    pts = L['labeledpos']
    nmov = len(localdirs)
    fly_id = []
    num_labels = np.zeros([nmov,])
    for ndx in range(nmov):

        dstr = localdirs[ndx].split('/')
        fly_id.append(dstr[-3])
        curpts = np.array(L[pts[0,ndx]])
        frames = np.where(np.invert(np.all(np.isnan(curpts[:, :, :]), axis=(1, 2))))[0]
        num_labels[ndx] = len(frames)
    ##

    ufly = list(set(fly_id))
    fly_labels = np.zeros([len(ufly)])
    for ndx in range(len(ufly)):
        fly_ndx = [i for i, x in enumerate(fly_id) if x == ufly[ndx]]
        fly_labels[ndx] = np.sum(num_labels[fly_ndx])

    ##

    folds = 3
    lbls_fold = int(old_div(np.sum(num_labels),folds))

    imbalance = True
    while imbalance:
        per_fold = np.zeros([folds])
        fly_fold = np.zeros(len(ufly))
        for ndx in range(len(ufly)):
            done = False
            curfold = np.random.randint(folds)
            while not done:
                if per_fold[curfold]>lbls_fold:
                    curfold = (curfold+1)%folds
                else:
                    fly_fold[ndx] = curfold
                    per_fold[curfold] += fly_labels[ndx]
                    done = True
        imbalance = (per_fold.max()-per_fold.min())>(old_div(lbls_fold,3))
    print(per_fold)

##
    for ndx in range(folds):
        curvaldatafilename = os.path.join(conf.cachedir,conf.valdatafilename + '_fold_{}'.format(ndx))
        fly_val = np.where(fly_fold==ndx)[0]
        isval = []
        for ndx in range(len(fly_val)):
            isval += [i for i, x in enumerate(fly_id) if x == ufly[fly_val[ndx]]]

        with open(curvaldatafilename,'w') as f:
            pickle.dump([isval,localdirs,seldirs],f)

##

def createvaldataRoian(conf):
    ##
    from roianMouseConfig import conf
    ##
    L  = h5py.File(conf.labelfile)
    localdirs,seldirs = multiResData.find_local_dirs(conf)

    ##
    pts = L['labeledpos']
    nmov = len(localdirs)
    fly_id = []
    date_id = []
    num_labels = np.zeros([nmov,])
    for ndx in range(nmov):

        dstr = localdirs[ndx].split('/')[-1]
        aa = re.search('experiment_(\d+)_(\d+)T(\d+).*_Mouse(\d+)', dstr).groups()
        fly_id.append('_'.join([aa[x] for x in [0,1,3]]))
        date_id.append(aa[1])
        curpts = np.array(L[pts[0,ndx]])
        frames = np.where(np.invert(np.all(np.isnan(curpts[:, :, :]), axis=(1, 2))))[0]
        num_labels[ndx] = len(frames)
    ##

    ufly = list(set(fly_id))
    fly_labels = np.zeros([len(ufly)])
    for ndx in range(len(ufly)):
        fly_ndx = [i for i, x in enumerate(fly_id) if x == ufly[ndx]]
        fly_labels[ndx] = np.sum(num_labels[fly_ndx])

    ##

    folds = 3
    lbls_fold = int(old_div(np.sum(num_labels),folds))

    imbalance = True
    while imbalance:
        per_fold = np.zeros([folds])
        fly_fold = np.zeros(len(ufly))
        for ndx in range(len(ufly)):
            done = False
            curfold = np.random.randint(folds)
            while not done:
                if per_fold[curfold]>lbls_fold:
                    curfold = (curfold+1)%folds
                else:
                    fly_fold[ndx] = curfold
                    per_fold[curfold] += fly_labels[ndx]
                    done = True
        imbalance = (per_fold.max()-per_fold.min())>(old_div(lbls_fold,2))
    print(per_fold)

##
    allisval = []
    for ndx in range(folds):
        curvaldatafilename = os.path.join(conf.cachedir,conf.valdatafilename + '_fold_{}'.format(ndx))
        fly_val = np.where(fly_fold==ndx)[0]
        isval = []
        for indx in range(len(fly_val)):
            isval += [i for i, x in enumerate(fly_id) if x == ufly[fly_val[indx]]]
        allisval.append(isval)
        with open(curvaldatafilename,'w') as f:
            pickle.dump([isval,localdirs,seldirs],f)
        io.savemat('/groups/branson/bransonlab/mayank/PoseTF/data/roian/valSplits.mat', {'split': allisval})
##
def createvaldataJay():
    ##
    from jayMouseConfig import sideconf as conf
    from jayMouseConfig import frontconf
    ##
    L  = h5py.File(conf.labelfile)
    localdirs,seldirs = multiResData.find_local_dirs(conf)

    ##
    pts = L['labeledpos']
    nmov = len(localdirs)
    fly_id = []
    date_id = []
    num_labels = np.zeros([nmov,])
    for ndx in range(nmov):

        dstr = localdirs[ndx].split('/')
        if re.search('M(\d+)_(\d{8})_', dstr[-2]):
            mm = re.search('M(\d+)_(\d{8})_', dstr[-2]).groups()
            aa = '_'.join(mm)
        else:
            mm = re.search('M(\d+)_(\d{8})_', dstr[-1]).groups()
            aa = '_'.join(mm)
        fly_id.append(aa)
        date_id.append(aa[1])
        curpts = np.array(L[pts[0,ndx]])
        frames = np.where(np.invert(np.all(np.isnan(curpts[:, :, :]), axis=(1, 2))))[0]
        num_labels[ndx] = len(frames)
    ##

    ufly = list(set(fly_id))
    fly_labels = np.zeros([len(ufly)])
    for ndx in range(len(ufly)):
        fly_ndx = [i for i, x in enumerate(fly_id) if x == ufly[ndx]]
        fly_labels[ndx] = np.sum(num_labels[fly_ndx])

    ##

    folds = 3
    lbls_fold = int(old_div(np.sum(num_labels),folds))

    imbalance = True
    while imbalance:
        per_fold = np.zeros([folds])
        fly_fold = np.zeros(len(ufly))
        for ndx in range(len(ufly)):
            done = False
            curfold = np.random.randint(folds)
            while not done:
                if per_fold[curfold]>lbls_fold:
                    curfold = (curfold+1)%folds
                else:
                    fly_fold[ndx] = curfold
                    per_fold[curfold] += fly_labels[ndx]
                    done = True
        imbalance = (per_fold.max()-per_fold.min())>(old_div(lbls_fold,2.5))
    print(per_fold)

##
    allisval = []
    localdirsf,seldirsf = multiResData.find_local_dirs(frontconf)

    for ndx in range(folds):
        curvaldatafilename = os.path.join(conf.cachedir,conf.valdatafilename + '_fold_{}'.format(ndx))
        curvaldatafilenamef = os.path.join(frontconf.cachedir,frontconf.valdatafilename + '_fold_{}'.format(ndx))
        fly_val = np.where(fly_fold==ndx)[0]
        isval = []
        for indx in range(len(fly_val)):
            isval += [i for i, x in enumerate(fly_id) if x == ufly[fly_val[indx]]]
        allisval.append(isval)
        with open(curvaldatafilename,'w') as f:
            pickle.dump([isval,localdirs,seldirs],f)
        with open(curvaldatafilenamef,'w') as f:
            pickle.dump([isval,localdirsf,seldirsf],f)
        io.savemat('/groups/branson/bransonlab/mayank/PoseTF/data/jayMouse/valSplits.mat', {'split': allisval})
##

def trainfold(conffile,curfold,curgpu,batch_size,onlydb=False,confname='conf'):

    imp_mod = importlib.import_module(conffile)
    conf = imp_mod.__dict__[confname]
    if batch_size>0:
        conf.batch_size = batch_size

    ext = '_fold_{}'.format(curfold)
    conf.valdatafilename = conf.valdatafilename + ext
    conf.trainfilename = conf.trainfilename + ext
    conf.valfilename = conf.valfilename + ext
    conf.fulltrainfilename += ext
    conf.baseoutname = conf.baseoutname + ext
    conf.mrfoutname += ext
    conf.fineoutname += ext
    conf.baseckptname += ext
    conf.mrfckptname += ext
    conf.fineckptname += ext
    conf.basedataname += ext
    conf.finedataname += ext
    conf.mrfdataname += ext


    if not os.path.exists(os.path.join(conf.cachedir,conf.trainfilename+'.tfrecords')):
        _, localdirs, seldirs = multiResData.load_val_data(conf)
        for ndx, curl in enumerate(localdirs):
            if not os.path.exists(curl):
                print(curl + ' {} doesnt exist!!!!'.format(ndx))
                return

        multiResData.create_tf_record_from_lbl(conf, True)
    if onlydb:
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = curgpu
    tf.reset_default_graph()
    self = PoseTrain.PoseTrain(conf)
    self.baseTrain(restore=True,trainPhase=True,trainType=0)
    tf.reset_default_graph()
    self.mrfTrain(restore=True,trainType=0)

def trainfold_fine(conffile,curfold,curgpu,batch_size,confname='conf'):

    imp_mod = importlib.import_module(conffile)
    conf = imp_mod.__dict__[confname]
    if batch_size>0:
        conf.batch_size = batch_size

    ##

    ext = '_fold_{}'.format(curfold)
    conf.valdatafilename = conf.valdatafilename + ext
    conf.trainfilename = conf.trainfilename + ext
    conf.valfilename = conf.valfilename + ext
    conf.fulltrainfilename += ext
    conf.baseoutname = conf.baseoutname + ext
    conf.mrfoutname += ext
    conf.fineoutname += ext
    conf.baseckptname += ext
    conf.mrfckptname += ext
    conf.fineckptname += ext
    conf.basedataname += ext
    conf.finedataname += ext
    conf.mrfdataname += ext

    os.environ['CUDA_VISIBLE_DEVICES'] = curgpu
    tf.reset_default_graph()
    self = PoseTrain.PoseTrain(conf)
    self.fineTrain(restore=True,trainPhase=True,trainType=0)



def classifyfold(conffile,curfold,curgpu,batch_size,
                 redo,outdir,confname='conf',outtype = 2,
                 extra_str=None):

    imp_mod = importlib.import_module(conffile)
    conf = imp_mod.__dict__[confname]
    if batch_size>0:
        conf.batch_size = batch_size

    ext = '_fold_{}'.format(curfold)
    conf.valdatafilename = conf.valdatafilename + ext
    conf.trainfilename = conf.trainfilename + ext
    conf.valfilename = conf.valfilename + ext
    conf.fulltrainfilename += ext
    conf.baseoutname = conf.baseoutname + ext
    conf.mrfoutname += ext
    conf.fineoutname += ext
    conf.baseckptname += ext
    conf.mrfckptname += ext
    conf.fineckptname += ext
    conf.basedataname += ext
    conf.finedataname += ext
    conf.mrfdataname += ext

    isval,localdirs,seldirs = multiResData.load_val_data(conf)
    for ndx,curl in enumerate(localdirs):
        if not os.path.exists(curl):
            print(curl + ' {} doesnt exist!!!!'.format(ndx))
            return

    os.environ['CUDA_VISIBLE_DEVICES'] = curgpu


    max_chunk_size = 1000
    self = PoseTools.create_network(conf, outtype)
    sess = tf.Session()
    PoseTools.init_network(self, sess, outtype)

    for ndx in range(len(localdirs)):
        if not isval.count(ndx):
            continue
        mname,_ = os.path.splitext(os.path.basename(localdirs[ndx]))
        oname = re.sub('!','__',conf.getexpname(localdirs[ndx]))

        if extra_str is not None:
            oname += '_'+extra_str
        print(oname)
        pname = os.path.join(outdir , oname)

        # detect

        if redo or not (os.path.isfile(pname + '.mat') or os.path.isfile(pname + '.h5')):

            cap = cv2.VideoCapture(localdirs[ndx])
            height = int(cap.get(cvc.FRAME_HEIGHT))
            width = int(cap.get(cvc.FRAME_WIDTH))
            orig_crop_loc = conf.cropLoc[(height,width)]
            nframes = int(cap.get(cvc.FRAME_COUNT))
            nblocks = int(math.ceil(float(nframes)/max_chunk_size))
            cap.release()

            out_sz = [0,0]
            out_sz[0] = height//conf.pool_scale//conf.rescale
            out_sz[1] = width//conf.pool_scale//conf.rescale
            with h5py.File(pname+'.h5','w') as f:
                f.create_dataset('expname', data=localdirs[ndx])
                for cur_b in range(nblocks):
                    startat = cur_b*max_chunk_size
                    stopat = min((cur_b+1)*max_chunk_size,nframes)
                    frames_read = stopat-startat
                    predList = PoseTools.classify_movie(conf, localdirs[ndx], outtype,
                                                        self, sess, max_frames=frames_read,
                                                        start_at=startat)
                    scale_fac = conf.pool_scale*conf.rescale
                    crop_loc = [old_div(x,scale_fac) for x in orig_crop_loc]
                    end_pad = [old_div(height,scale_fac)-crop_loc[0]-old_div(conf.imsz[0],scale_fac),
                               old_div(width,scale_fac)-crop_loc[1]-old_div(conf.imsz[1],scale_fac)]
                    pp = [(0,0),(crop_loc[0],end_pad[0]),(crop_loc[1],end_pad[1]),(0,0),(0,0)]
                    predScores = np.pad(predList[1],pp,mode='constant',constant_values=-1.)

                    if cur_b == 0:
                        locs_o = f.create_dataset('locs', (nframes, conf.n_classes, 2, 2))
                        scores_o = f.create_dataset('scores', (nframes, predScores.shape[1],
                                                               predScores.shape[2], conf.n_classes, 2))

                    predLocs = predList[0]
                    predLocs[:,:,:,0] += orig_crop_loc[1]
                    predLocs[:,:,:,1] += orig_crop_loc[0]

                    locs_o[startat:stopat,...] = predLocs
                    scores_o[startat:stopat,...] = predScores

            # io.savemat(pname + '.mat',{'locs':predLocs,'scores':predScores[...,0],'expname':localdirs[ndx]})
            print('Detecting:%s'%oname)

def classifyfold_fine(conffile,curfold,curgpu,batch_size,
                 redo,outdir,results_dir,confname='conf',extra_str=None):

    imp_mod = importlib.import_module(conffile)
    conf = imp_mod.__dict__[confname]
    if batch_size>0:
        conf.batch_size = batch_size

    ext = '_fold_{}'.format(curfold)
    conf.valdatafilename = conf.valdatafilename + ext
    conf.trainfilename = conf.trainfilename + ext
    conf.valfilename = conf.valfilename + ext
    conf.fulltrainfilename += ext
    conf.baseoutname = conf.baseoutname + ext
    conf.mrfoutname += ext
    conf.fineoutname += ext
    conf.baseckptname += ext
    conf.mrfckptname += ext
    conf.fineckptname += ext
    conf.basedataname += ext
    conf.finedataname += ext
    conf.mrfdataname += ext

    isval,localdirs,seldirs = multiResData.load_val_data(conf)

    os.environ['CUDA_VISIBLE_DEVICES'] = curgpu

    max_chunk_size = 5000
    outtype = 3
    self = PoseTools.create_network(conf, outtype)
    sess = tf.Session()
    PoseTools.init_network(self, sess, outtype)

    for ndx in range(len(localdirs)):
        if not isval.count(ndx):
            continue
        mname,_ = os.path.splitext(os.path.basename(localdirs[ndx]))
        ename = re.sub('!','__',conf.getexpname(localdirs[ndx]))
        if extra_str is not None:
            oname = ename+ '_'+extra_str
        else:
            oname = ename
        print(oname)


        pname = os.path.join(outdir , oname) + '_fine'
        rname = os.path.join(results_dir,ename) + '.mat'
        S = h5py.File(rname,'r')
        view = conf.view
        locs = np.array(S[S['R'][view,0]]['final_locs'])
        locs = np.transpose(locs,[2,1,0])
        # detect
        if redo or not (os.path.isfile(pname + '.h5')):
            cap = cv2.VideoCapture(localdirs[ndx])
            height = int(cap.get(cvc.FRAME_HEIGHT))
            width = int(cap.get(cvc.FRAME_WIDTH))
            orig_crop_loc = conf.cropLoc[(height,width)]
            nframes = int(cap.get(cvc.FRAME_COUNT))
            nblocks = int(math.ceil(float(nframes)/max_chunk_size))
            cap.release()

            locs[:,:,0] -= orig_crop_loc[1]
            locs[:,:,1] -= orig_crop_loc[0]

            with h5py.File(pname+'.h5','w') as f:
                f.create_dataset('expname', data=localdirs[ndx])
                locs_o = f.create_dataset('locs',  (nframes,conf.n_classes,2))

                for cur_b in range(nblocks):
                    startat = cur_b*max_chunk_size
                    stopat = min((cur_b+1)*max_chunk_size,nframes)
                    frames_read = stopat-startat

                    predLocs = PoseTools.classify_movie_fine(conf, localdirs[ndx], locs,
                                                             self, sess, max_frames=frames_read,
                                                             start_at=startat)

                    predLocs[:,:,0] += orig_crop_loc[1]
                    predLocs[:,:,1] += orig_crop_loc[0]
                    locs_o[startat:stopat,...] = predLocs
            # io.savemat(pname + '.mat',{'locs':predLocs,'scores':predScores[...,0],'expname':localdirs[ndx]})
            print('\nFine Detection done for :%s'%oname)

def createCVMat(conffile):
    imp_mod = importlib.import_module(conffile)
##
    allisval = []
    for curfold in range(3):
        conf = copy.deepcopy(imp_mod.conf)
        ext = '_fold_{}'.format(curfold)
        conf.valdatafilename = conf.valdatafilename + ext

        isval,localdirs,seldirs = multiResData.load_val_data(conf)
        allisval.append(isval)
    return allisval
##

def checkDB(conffile,confname='conf'):
    imp_mod = importlib.import_module(conffile)
    conf = imp_mod.__dict__[confname]
##
    curfold = 0

    ##
    ext = '_fold_{}'.format(curfold)
    conf.valdatafilename = conf.valdatafilename + ext
    conf.trainfilename = conf.trainfilename + ext
    conf.valfilename = conf.valfilename + ext
    conf.fulltrainfilename += ext
    conf.baseoutname = conf.baseoutname + ext
    conf.mrfoutname += ext
    conf.fineoutname += ext
    conf.baseckptname += ext
    conf.mrfckptname += ext
    conf.fineckptname += ext
    conf.basedataname += ext
    conf.finedataname += ext
    conf.mrfdataname += ext

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    tf.reset_default_graph()
    self = PoseTrain.PoseTrain(conf)
    self.createPH()
    self.createFeedDict()
    trainPhase = True
    trainType = 0
    self.feed_dict[self.ph['phase_train_base']] = trainPhase
    self.feed_dict[self.ph['keep_prob']] = 0.5
    self.trainType = trainType
    self.openDBs()
    sess = tf.InteractiveSession()
    self.createCursors(sess)
    ##
    for ndx in range(np.random.randint(1,40)):
        self.updateFeedDict(self.DBType.Train, sess=sess, distort=True)
    xs = self.xs
    selex = np.random.randint(conf.batch_size)
    plt.cla()
    plt.imshow(xs[selex,0,...],cmap='gray')
    locs = self.locs
    plt.scatter(locs[selex,:,0],locs[selex,:,1])
##
if __name__ == "__main__":
    main(sys.argv)

def duplicatesJan():
    ##
    from janLegConfig import conf
    import collections
    localdirs,seldirs = multiResData.find_local_dirs(conf)
    gg = [conf.getexpname(x) for x in localdirs]
    mm = [item for item, count in list(collections.Counter(gg).items()) if count > 1]
    L = h5py.File(conf.labelfile)
    pts = L['labeledpos']
    for ndx in range(len(mm)):
        aa = [idx for idx, x in enumerate(gg) if x == mm[ndx]]
        print(mm[ndx])
        for curndx in aa:
            curpts = np.array(L[pts[0, curndx]])
            frames = np.where(np.invert( np.all(np.isnan(curpts[:,:,:]),axis=(1,2))))[0]
            print(curndx,frames.size)
##


