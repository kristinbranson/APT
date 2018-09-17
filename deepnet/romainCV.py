from __future__ import division
from __future__ import print_function
from builtins import range
from past.utils import old_div
import tensorflow as tf
from multiResData import *
import h5py
import pickle
import sys
import os
import PoseTrain
import multiResData
from romainLegConfig import side1conf as side1conf
from romainLegConfig import side2conf as side2conf
from romainLegConfig import bottomconf as bottomconf

##
def main(argv):
    view = int(argv[1])
    curfold = int(argv[2])
    curgpu = argv[3]
    if len(argv)>=5:
        batch_size = int(argv[4])
    else:
        batch_size = -1

    trainfold(view=view,curfold=curfold,curgpu=curgpu,batch_size=batch_size)

def trainfold(view,curfold,curgpu,batch_size):
    if view == 0:
        conf = side1conf
    elif view == 1:
        conf = side2conf
    elif view ==2:
        conf = bottomconf
    else:
        sys.stderr('Incorrect view %d'.format(view))

    if batch_size>0:
        conf.batch_size = batch_size

    createvaldata = False
    # conf.cachedir = os.path.join(conf.cachedir,'cross_val_fly')

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

    _,localdirs,seldirs = multiResData.load_val_data(conf)
    for ndx,curl in enumerate(localdirs):
        if not os.path.exists(curl):
            print(curl + ' {} doesnt exist!!!!'.format(ndx))
            return


    if not os.path.exists(os.path.join(conf.cachedir,conf.trainfilename+'.tfrecords')):
        multiResData.create_tf_record_from_lbl(conf, True)
    os.environ['CUDA_VISIBLE_DEVICES'] = curgpu
    tf.reset_default_graph()
    self = PoseTrain.PoseTrain(conf)
    self.baseTrain(restore=True,trainPhase=True,trainType=0)
    tf.reset_default_graph()
    self.mrfTrain(restore=True,trainType=0)


if __name__ == "__main__":
    main(sys.argv)
def createvaldata():
    ##
    conf = side1conf
    L  = h5py.File(conf.labelfile)
    localdirs,seldirs = multiResData.find_local_dirs(conf)
    localdirs2,seldirs2 = multiResData.find_local_dirs(side2conf)
    localdirsb,seldirsb = multiResData.find_local_dirs(bottomconf)

    ##
    pts = L['labeledpos']
    nmov = len(localdirs)
    fly_id = []
    num_labels = np.zeros([nmov,])
    for ndx in range(nmov):

        dstr = localdirs[ndx].split('/')[-2]
        fly_id.append(dstr)
        curpts = np.array(L[pts[0,ndx]])
        frames = np.where(np.invert(np.all(np.isnan(curpts[:, :, :]), axis=(1, 2))))[0]
        num_labels[ndx] = len(frames)
    ##

    ufly = list(set(fly_id[1:])) # keep movie 1 aside
    fly_labels = np.zeros([len(ufly)])
    for ndx in range(len(ufly)):
        fly_ndx = [i for i, x in enumerate(fly_id) if x == ufly[ndx]]
        fly_labels[ndx] = np.sum(num_labels[fly_ndx])

    ##

    folds = 2
    lbls_fold = int(old_div(np.sum(num_labels[1:]),folds)) # keep movie 1 common

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
        curvaldatafilename2 = os.path.join(side2conf.cachedir,side2conf.valdatafilename + '_fold_{}'.format(ndx))
        curvaldatafilenameb = os.path.join(bottomconf.cachedir,bottomconf.valdatafilename + '_fold_{}'.format(ndx))
        fly_val = np.where(fly_fold==ndx)[0]
        isval = []
        for ix in range(len(fly_val)):
            isval += [i for i, x in enumerate(fly_id) if x == ufly[fly_val[ix]]]
        with open(curvaldatafilename,'w') as f:
            pickle.dump([isval,localdirs,seldirs],f)
        with open(curvaldatafilename2,'w') as f:
            pickle.dump([isval,localdirs2,seldirs2],f)
        with open(curvaldatafilenameb,'w') as f:
            pickle.dump([isval,localdirsb,seldirsb],f)

##
def createViewFiles(view):
    ##
    if view == 0:
        conf = sideconf
    else:
        conf = frontconf
    conf.cachedir = os.path.join(conf.cachedir,'cross_val_fly')

    basepath = '/groups/branson/bransonlab/mayank/stephenCV/'
    for curfold in range(5):
        ext = '_fold_{}'.format(curfold)
        curvaldatafilename = conf.valdatafilename + ext
        with open(os.path.join(conf.cachedir,curvaldatafilename), 'r') as f:
            [isval, localdirs, seldirs] = pickle.load(f)

        with open(basepath+'view{}vids_fold_{}.txt'.format(view+1,curfold),'w') as f:
            for ndx,curf in enumerate(localdirs):
                if not isval.count(ndx):
                    continue
                pp = curf.split('/')
                startat = -1
                for mndx in range(len(pp)-1,0,-1):
                    if re.match('^fly_*\d+$',pp[mndx]):
                        startat = mndx
                        break
                cc = '/'.join(pp[startat:])
                cc = basepath + cc
                if not os.path.exists(cc):
                    continue
                f.write(cc+'\n')
##
def createSideValData():
    from stephenHeadConfig import sideconf as sideconf
    localdirs, seldirs = find_local_dirs(sideconf)

    folds = 5
    for ndx in range(folds):
        curvaldatafilename = os.path.join(conf.cachedir, conf.valdatafilename + '_fold_{}'.format(ndx))
        with open(curvaldatafilename, 'r') as f:
            [isval, _a,_b] = pickle.load(f)
        curvaldatafilename = os.path.join(sideconf.cachedir, sideconf.valdatafilename + '_fold_{}'.format(ndx))
        with open(curvaldatafilename, 'w') as f:
            pickle.dump([isval, localdirs, seldirs], f)


##
# folds = 5
# allisval = []
# for ndx in range(1,folds):
#     curvaldatafilename = os.path.join(conf.cachedir, conf.valdatafilename + '_fold_{}'.format(ndx))
#     with open(curvaldatafilename, 'r') as f:
#         [isval, localdirs,seldirs] = pickle.load(f)
#     allisval += isval

##
# folds = 5
# for ndx in range(0,folds):
#     curvaldatafilename = os.path.join(conf.cachedir, conf.valdatafilename + '_fold_{}'.format(ndx))
#     with open(curvaldatafilename, 'r') as f:
#         [isval, _localdirs,seldirs] = pickle.load(f)
#     with open(curvaldatafilename, 'w') as f:
#         pickle.dump([isval, localdirs,seldirs] ,f)
