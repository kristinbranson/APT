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
import tensorflow as tf
from stephenHeadConfig import sideconf as sideconf
from stephenHeadConfig import conf as frontconf
import argparse
import copy

##

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("-fold",dest="fold",
                      help="fold number",type=int,
                      required=True)
    parser.add_argument("-gpu",dest='gpunum',
                        help="GPU to use",required=True)
    parser.add_argument("-view",dest='view',type=int,
                        help="View,0 is side, 1 is front",required=True)
    parser.add_argument("-bs",dest='batch_size',type=int,
                        help="batch_size to use [optional]")
    parser.add_argument("-train",dest='train',action="store_true",
                        help="Train")
    parser.add_argument("-detect",dest='detect',action="store_true",
                        help="Detect")
    parser.add_argument("-fine_train",dest='fine_train',action="store_true",
                        help="Fine Train")
    parser.add_argument("-fine_detect",dest='fine_detect',action="store_true",
                        help="Fine Detect")
    parser.add_argument("-r",dest="redo",
                      help="if specified will recompute everything for detecting",
                      action="store_true")
    parser.add_argument("-o",dest="outdir",
                      help="temporary output directory to store intermediate computations")
    parser.add_argument("-results_dir",dest="results_dir",
                      help="Directory with tracking results")
    parser.add_argument("-train_size",dest="train_size",action='store_true',
                      help="Train size varying classifiers")

    args = parser.parse_args(argv[1:])

    view = args.view
    curfold = args.fold
    curgpu = args.gpunum
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = -1

    if args.train:
        trainfold(view=view,curfold=curfold,curgpu=curgpu,batch_size=batch_size)

    if args.fine_train:
        trainfold_fine(view=view, curfold=curfold, curgpu=curgpu, batch_size=batch_size)

    if args.fine_detect:
        classifyfold_fine(view=view, curfold=curfold, curgpu=curgpu, batch_size=batch_size,
                          redo=args.redo,outdir=args.outdir,results_dir=args.results_dir)

    if args.train_size:
        trainfold_size(view=view,curfold=curfold,curgpu=curgpu)

def trainfold(view,curfold,curgpu,batch_size):
    if view == 0:
        conf = sideconf
    else:
        conf = frontconf

    if batch_size>0:
        conf.batch_size = batch_size

    createvaldata = False
    conf.cachedir = os.path.join(conf.cachedir,'cross_val_fly_psz')

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


def trainfold_fine(view,curfold,curgpu,batch_size):
    if view == 0:
        conf = sideconf
    else:
        conf = frontconf

    if batch_size>0:
        conf.batch_size = batch_size

    conf.cachedir = os.path.join(conf.cachedir,'cross_val_fly_psz')
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

def classifyfold_fine(curfold,curgpu,batch_size,
                 redo,outdir,results_dir,view):

    if view == 0:
        conf = sideconf
        extra_str = '_side_fine'
        filename = '/groups/branson/bransonlab/mayank/stephenCV/view1vids_fold_{}.txt'.format(curfold)
        with open(filename, "r") as text_file:
            smovies = text_file.readlines()
        movies = [x.rstrip() for x in smovies]
    else:
        conf = frontconf
        extra_str = '_front_fine'
        filename = '/groups/branson/bransonlab/mayank/stephenCV/view2vids_fold_{}.txt'.format(curfold)
        with open(filename, "r") as text_file:
            smovies = text_file.readlines()
        movies = [x.rstrip() for x in smovies]

    if batch_size>0:
        conf.batch_size = batch_size

    conf.cachedir = os.path.join(conf.cachedir,'cross_val_fly_psz')

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

    outtype = 3
    self = PoseTools.create_network(conf, outtype)
    sess = tf.Session()
    PoseTools.init_network(self, sess, outtype)


    for ndx in range(len(movies)):
        mname,_ = os.path.splitext(os.path.basename(movies[ndx]))

        oname = re.sub('!','__',conf.getexpname(movies[ndx]))
        pname = os.path.join(outdir, oname + extra_str)
        print(oname)


        rname = os.path.join(results_dir,oname) + '.mat'
        if not os.path.exists(rname):
            print('{} {} doesnt exist'.format(ndx,rname))
            continue
        S = h5py.File(rname,'r')
        view = conf.view
        locs = np.array(S[S['R'][view,0]]['final_locs'])
        locs = np.transpose(locs,[2,1,0])
        # detect
        if redo or not (os.path.isfile(pname + '.h5')):
            cap = cv2.VideoCapture(movies[ndx])
            height = int(cap.get(cvc.FRAME_HEIGHT))
            width = int(cap.get(cvc.FRAME_WIDTH))
            orig_crop_loc = conf.cropLoc[(height,width)]

            locs[:,:,0] -= orig_crop_loc[1]
            locs[:,:,1] -= orig_crop_loc[0]

            predLocs = PoseTools.classify_movie_fine(conf,movies[ndx],locs,self,sess)

            predLocs[:,:,0] += orig_crop_loc[1]
            predLocs[:,:,1] += orig_crop_loc[0]

            with h5py.File(pname+'.h5','w') as f:
                f.create_dataset('locs',data=predLocs)
                f.create_dataset('expname', data=movies[ndx])
            # io.savemat(pname + '.mat',{'locs':predLocs,'scores':predScores[...,0],'expname':localdirs[ndx]})
            print('\nFine detection done for :%s'%oname)


def trainfold_size(view, curfold, curgpu):
    if view == 0:
        conf = sideconf
    else:
        conf = frontconf

    conf.cachedir = os.path.join(conf.cachedir, 'size_exp')
    tr_frac = np.array([0.1,0.25,0.5,0.75,0.9])

    ##

    ext = '_sz_{}'.format(int(tr_frac[curfold]*100))
    conf.trainfilename = conf.trainfilename + ext
    conf.baseoutname += ext
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
    self.baseTrain(restore=True, trainPhase=True, trainType=0)
    tf.reset_default_graph()
    self.mrfTrain(restore=True, trainType=0)


if __name__ == "__main__":
    main(sys.argv)


def create_val_data(conf):
    ##
    L  = h5py.File(conf.labelfile)
    localdirs,seldirs = find_local_dirs(conf)

    ##
    pts = L['labeledpos']
    nmov = len(localdirs)
    fly_num = np.zeros([nmov,])
    num_labels = np.zeros([nmov,])
    for ndx in range(nmov):
        f_str = re.search('fly_*(\d*)',localdirs[ndx])
        fly_num[ndx] = int(f_str.groups()[0])
        curpts = np.array(L[pts[0,ndx]])
        frames = np.where(np.invert(np.all(np.isnan(curpts[:, :, :]), axis=(1, 2))))[0]
        num_labels[ndx] = len(frames)
    ##

    ufly,uidx = np.unique(fly_num,return_inverse=True)
    fly_labels = np.zeros([len(ufly)])
    for ndx in range(len(ufly)):
        fly_labels[ndx] = np.sum(num_labels[uidx==ndx])

    ##

    folds = 5
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
        isval = np.where(np.in1d(uidx,fly_val))[0].tolist()
        with open(curvaldatafilename,'w') as f:
            pickle.dump([isval,localdirs,seldirs],f)


def update_dirnames(conf):
##

    localdirs, seldirs = find_local_dirs(conf)

    newname = [False]*len(localdirs)
    for ndx in range(len(localdirs)):
        if localdirs[ndx].find('fly_450_to_452_26_9_1')>0:
            localdirs[ndx] = localdirs[ndx][:50] + 'fly_450_to_452_26_9_16norpAkirPONMNchr' + localdirs[ndx][72:]

    newname = [False]*len(localdirs)
    for ndx in range(len(localdirs)):
        if localdirs[ndx].find('fly_453_to_457_27_9_16')>0:
            localdirs[ndx] = localdirs[ndx][:50] + 'fly_453_to_457_27_9_16_norpAkirPONMNchr' + localdirs[ndx][72:]

    for ndx,curl in enumerate(localdirs):
        if not os.path.exists(curl):
            print(curl + ' {} doesnt exist!!!!'.format(ndx))
    return localdirs


##
def createViewFiles(view):
    ##
    if view == 0:
        conf = sideconf
    else:
        conf = frontconf
    conf.cachedir = os.path.join(conf.cachedir,'cross_val_fly_psz')

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


def size_exp_datasets():
    ##
    def _int64_feature(value):
        if not isinstance(value, (list, np.ndarray)):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def _float_feature(value):
        if not isinstance(value, (list, np.ndarray)):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    ##
    from stephenHeadConfig import conf as frontconf
    from stephenHeadConfig import sideconf
    
    conf = []
    conf.append(copy.deepcopy(frontconf))
    conf.append(copy.deepcopy(sideconf))
##
    ims = [[],[]]
    exp_ids = [[],[]]
    ts = [[],[]]
    locs = [[],[]]
    for view in range(2):
        orig_train_file_name = os.path.join(conf[view].cachedir, 'cross_val_fly_psz',
                                            conf[view].trainfilename + '_fold_0')
        in_tf = tf.python_io.tf_record_iterator(orig_train_file_name + '.tfrecords')
        for string_record in in_tf:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            exp_id_in = int(example.features.feature['expndx']
                         .float_list
                         .value[0])
            ts_in = int(example.features.feature['ts']
                        .float_list
                        .value[0])
            img_string = (example.features.feature['image_raw']
                          .bytes_list
                          .value[0])
            img_1d = np.fromstring(img_string, dtype=np.uint8)
            reconstructed_img = img_1d.reshape((conf[view].imsz[0], conf[view].imsz[1], conf[view].imgDim))
            locs_in = np.reshape(np.array(example.features.feature['locs'].float_list.value),
                                 [conf[view].n_classes,2])

            ims[view].append(reconstructed_img)
            exp_ids[view].append(exp_id_in)
            ts[view].append(ts_in)
            locs[view].append(locs_in)
    ##
    tr_frac = np.array([0.1,0.25,0.5,0.75,0.9])
    u_exp, c_exp = np.unique(np.array(exp_ids[0]), return_counts=True)
    perm = np.random.permutation(len(u_exp))
    u_exp = u_exp[perm]
    c_exp = c_exp[perm]
    balance = 1
    count = 0
    while balance>0.05 and count<10:
        perm = np.random.permutation(len(u_exp))
        u_exp = u_exp[perm]
        c_exp = c_exp[perm]
        gg = np.cumsum(c_exp)/c_exp.sum()
        ids = (tr_frac*len(u_exp)).astype('int')
        balance = np.abs(gg[ids]-tr_frac).max()
        print('{:.2f}'.format(balance))
        count += 1
    ##

    for sz in range(len(tr_frac)):
        last_ndx = int(tr_frac[sz]*len(u_exp))
        for view in range(2):
            train_file_name = os.path.join(conf[view].cachedir, 'size_exp',
                                         conf[view].trainfilename + '_sz_{}'.format(int(tr_frac[sz]*100)))
            cur_tf = tf.python_io.TFRecordWriter(train_file_name + '.tfrecords')

            for e_idx in range(last_ndx):
                idx = np.where(exp_ids[view]==u_exp[e_idx])[0]
                for cur_idx in idx:
                    cur_img = ims[view][cur_idx]
                    rows = cur_img.shape[0]
                    cols = cur_img.shape[1]
                    if np.ndim(cur_img) > 2:
                        depth = cur_img.shape[2]
                    else:
                        depth = 1

                    cur_loc = locs[view][cur_idx]
                    cur_exp = exp_ids[view][cur_idx]
                    cur_ts = ts[view][cur_idx]
                    image_raw = cur_img.tostring()
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'height': _int64_feature(rows),
                        'width': _int64_feature(cols),
                        'depth': _int64_feature(depth),
                        'locs': _float_feature(cur_loc.flatten()),
                        'expndx': _float_feature(cur_exp),
                        'ts': _float_feature(cur_ts),
                        'image_raw': _bytes_feature(image_raw)}))
                    cur_tf.write(example.SerializeToString())


            cur_tf.close()


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
