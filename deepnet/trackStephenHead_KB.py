from __future__ import division
from __future__ import print_function

#!/usr/bin/python
from builtins import range
from past.utils import old_div
import os
import re
import glob
import sys
import argparse
from subprocess import call
import stat
import h5py
import hdf5storage
import APT_interface as apt
import numpy as np
import movies
import math
import PoseTools

# default_net_name = 'pose_unet_full_20180302'
default_net_name = 'deepnet'
crop_reg_file = '/groups/branson/bransonlab/mayank/stephen_copy/crop_regression_params.mat'
# lbl_file = '/groups/branson/bransonlab/mayank/stephen_copy/apt_cache/sh_trn4523_gtcomplete_cacheddata_bestPrms20180920_retrain20180920T123534_withGTres.lbl'
# name = 'stephen_20181029'
#name = 'stephen_20181102_newlabels'
#lbl_file = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4879_gtcomplete_cacheddata_dlstripped.lbl'
#name = 'stephen_20181115'
lbl_file = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4992_gtcomplete_cacheddata_dlstripped.lbl'
#name = 'stephen_20190109'
name = 'stephen_20190126'
crop_size = [[230,350],[350,350]]
cache_dir = '/groups/branson/bransonlab/mayank/stephen_copy/apt_cache'
model_type = 'mdn'
#bodylblfile = '/groups/branson/bransonlab/mayank/stephen_copy/fly2BodyAxis_lookupTable_Ben.csv'
bodylblfile = '/groups/huston/hustonlab/flp-chrimson_experiments/fly2BodyAxis_lookupTable_Ben.csv'

#    defaulttrackerpath = "/groups/branson/home/bransonk/tracking/code/poseTF/matlab/compute3Dfrom2D/for_redistribution_files_only/run_compute3Dfrom2D.sh"
#    defaultmcrpath = "/groups/branson/bransonlab/projects/olympiad/MCR/v91"
# defaulttrackerpath = "/groups/branson/bransonlab/mayank/PoseTF/matlab/compiled/run_compute3Dfrom2D_compiled.sh"
# defaultmcrpath = "/groups/branson/bransonlab/mayank/MCR/v92"

defaulttrackerpath = "/groups/branson/bransonlab/mayank/stephen_copy/apt_cache/compiled_20181219/run_compute3Dfrom2D_compiled.sh"
defaultmcrpath = "/groups/branson/bransonlab/mayank/MCR/v94"

def get_crop_locs(lblfile,view,height,width):
    # everything is in matlab indexing
    bodylbl = apt.loadmat(lblfile)
    try:
        lsz = np.array(bodylbl['labeledpos']['size'])
        curpts = np.nan * np.ones(lsz).flatten()
        idx = np.array(bodylbl['labeledpos']['idx']) - 1
        val = np.array(bodylbl['labeledpos']['val'])
        curpts[idx] = val
        curpts = np.reshape(curpts, np.flipud(lsz))
    except IndexError:
        if bodylbl['labeledpos'].ndim == 3:
            curpts = np.array(bodylbl['labeledpos'])
            curpts = np.transpose(curpts, [2, 1, 0])
        else:
            if hasattr(bodylbl['labeledpos'][0],'idx'):
                lsz = np.array(bodylbl['labeledpos'][0].size)
                curpts = np.nan * np.ones(lsz).flatten()
                idx = np.array(bodylbl['labeledpos'][0].idx) - 1
                val = np.array(bodylbl['labeledpos'][0].val)
                curpts[idx] = val
                curpts = np.reshape(curpts, np.flipud(lsz))
            else:
                curpts = np.array(bodylbl['labeledpos'][0])
                curpts = np.transpose(curpts, [2, 1, 0])
    neck_locs = curpts[0, :, 5 + 10 * view]
    reg_params = apt.loadmat(crop_reg_file)
    x_reg = reg_params['reg_view{}_x'.format(view + 1)]
    y_reg = reg_params['reg_view{}_y'.format(view + 1)]
    x_left = int(round(x_reg[0] + x_reg[1] * neck_locs[0]))
    x_left = 1 if x_left < 1 else x_left
    x_right = x_left + crop_size[view][0] -1
    if x_right > width:
        x_left = width - crop_size[view][0] +1
        x_right = width
    y_top = int(round(y_reg[0] + y_reg[1] * neck_locs[1]))
    y_top = 1 if y_top < 1 else y_top
    y_bottom = y_top + crop_size[view][1] - 1
    if y_bottom > height:
        y_bottom = height
        y_top = height - crop_size[view][1] + 1
    return [x_left,x_right, y_top, y_bottom]


def classify_movie(mov_file, pred_fn, conf, crop_loc):
    cap = movies.Movie(mov_file)
    sz = (cap.get_height(), cap.get_width())
    n_frames = int(cap.get_n_frames())
    bsize = conf.batch_size
    flipud = False

    pred_locs = np.zeros([n_frames, conf.n_classes, 2])
    pred_ulocs = np.zeros([n_frames, conf.n_classes, 2])
    preds = np.zeros([n_frames,int(conf.imsz[0]//conf.rescale),int(conf.imsz[1]//conf.rescale),conf.n_classes])
    pred_locs[:] = np.nan
    uconf = np.zeros([n_frames, conf.n_classes])

    to_do_list = []
    for cur_f in range(0, n_frames):
        to_do_list.append([cur_f, 0])

    n_list = len(to_do_list)
    n_batches = int(math.ceil(float(n_list) / bsize))
    cc = [c-1 for c in crop_loc]
    for cur_b in range(n_batches):
        cur_start = cur_b * bsize
        ppe = min(n_list - cur_start, bsize)
        all_f = apt.create_batch_ims(to_do_list[cur_start:(cur_start+ppe)], conf,
                                 cap, flipud, [None], crop_loc=cc)

        # base_locs, hmaps = pred_fn(all_f)
        ret_dict = pred_fn(all_f)
        base_locs = ret_dict['locs']
        hmaps = ret_dict['hmaps']
        if model_type == 'mdn':
            uconf_cur = ret_dict['conf_unet']
            ulocs_cur = ret_dict['locs_unet']
        else:
            uconf_cur = ret_dict['conf']
            ulocs_cur = ret_dict['locs']

        for cur_t in range(ppe):
            cur_entry = to_do_list[cur_t + cur_start]
            cur_f = cur_entry[0]
            xlo, xhi, ylo, yhi = crop_loc
            base_locs_orig = base_locs[cur_t,...].copy()
            base_locs_orig[:, 0] += xlo
            base_locs_orig[:, 1] += ylo
            pred_locs[cur_f, :, :] = base_locs_orig[ ...]
            u_locs_orig = ulocs_cur[cur_t,...].copy()
            u_locs_orig[:, 0] += xlo
            u_locs_orig[:, 1] += ylo
            pred_ulocs[cur_f, :, :] = u_locs_orig[ ...]
            preds[cur_f,...] = hmaps[cur_t,...]
            uconf[cur_f,...] = uconf_cur[cur_t,...]

        if cur_b % 20 == 19:
            sys.stdout.write('.')
        if cur_b % 400 == 399:
            sys.stdout.write('\n')

    cap.close()
    return pred_locs,preds, pred_ulocs, uconf

def getexpname(dirname):
    dirname = os.path.normpath(dirname)
    dir_parts = dirname.split(os.sep)
    expname = dir_parts[-6] + "!" + dir_parts[-3] + "!" + dir_parts[-1][-10:-4]
    return expname

def update_conf(conf):
    conf.normalize_img_mean = False
    conf.adjust_contrast = True
    conf.dl_steps = 60000

def main(argv):


    parser = argparse.ArgumentParser()
    parser.add_argument("-s",dest="sfilename",
                      help="text file with list of side view videos",
                      required=True)
    parser.add_argument("-f",dest="ffilename",
                      help="text file with list of front view videos. The list of side view videos and front view videos should match up",
                      required=True)
    parser.add_argument("-d",dest="dltfilename",
                      help="text file with list of DLTs, one per fly as 'flynum,/path/to/dltfile'",
                      required=True)
    parser.add_argument("-body_lbl",dest="bodylabelfilename",
                      help="text file with list of body-label files, one per fly as 'flynum,/path/to/body_label.lbl'",
                      default=bodylblfile)
    parser.add_argument("-net",dest="net_name",
                      help="Name of the net to use for tracking",
                      default=default_net_name)
    parser.add_argument("-o",dest="outdir",
                      help="temporary output directory to store intermediate computations",
                      required=True)
    parser.add_argument("-r",dest="redo",
                      help="if specified will recompute everything",
                      action="store_true")
    parser.add_argument("-rt",dest="redo_tracking",
                      help="if specified will only recompute tracking",
                      action="store_true")
    parser.add_argument("-gpu",dest='gpunum',type=int,
                        help="GPU to use [optional]")
    parser.add_argument("-makemovie",dest='makemovie',
                        help="if specified will make results movie",action="store_true")
    parser.add_argument("-trackerpath",dest='trackerpath',
                        help="Absolute path to the compiled MATLAB tracker script run_compute3Dfrom2D.sh",
                        default=defaulttrackerpath)
    parser.add_argument("-mcrpath",dest='mcrpath',
                        help="Absolute path to MCR",
                        default=defaultmcrpath)
    parser.add_argument("-ncores",dest="ncores",
                        help="Number of cores to assign to each MATLAB tracker job", type=int,
                        default=1)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-only_detect",dest='detect',action="store_true",
                        help="Do only the detection part of tracking which requires GPU")
    group.add_argument("-only_track",dest='track',action="store_true",
                        help="Do only the tracking part of the tracking which requires MATLAB")

    args = parser.parse_args(argv)
    if args.redo is None:
        args.redo = False
    if args.redo_tracking is None:
        args.redo_tracking= False

    if args.detect is False and args.track is False: 
        args.detect = True
        args.track = True
    
    args.outdir = os.path.abspath(args.outdir)
    
    with open(args.sfilename, "r") as text_file:
        smovies = text_file.readlines()
    smovies = [x.rstrip() for x in smovies]
    with open(args.ffilename, "r") as text_file:
        fmovies = text_file.readlines()
    fmovies = [x.rstrip() for x in fmovies]

    print(smovies)
    print(fmovies)
    print(len(smovies))
    print(len(fmovies))


    if len(smovies) != len(fmovies):
        print("Side and front movies must match")
        raise exit(0)

    if args.track:
        # read in dltfile
        dltdict = {}
        f = open(args.dltfilename,'r')
        for l in f:
            lparts = l.split(',')
            if len(lparts) != 2:
                print("Error splitting dlt file line %s into two parts"%l)
                raise exit(0)
            dltdict[float(lparts[0])] = lparts[1].strip()
        f.close()
        
        # compiled matlab command
        matscript = args.trackerpath + " " + args.mcrpath

    if args.detect:
        import numpy as np
        import tensorflow as tf
        from scipy import io
        from cvc import cvc
        import localSetup
        import PoseTools
        import multiResData
        import cv2
        import PoseUNet

        for ff in smovies+fmovies:
            if not os.path.isfile(ff):
                print("Movie %s not found"%(ff))
                raise exit(0)
        if args.gpunum is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        bodydict = {}
        f = open(args.bodylabelfilename,'r')
        for l in f:
            lparts = l.split(',')
            if len(lparts) != 2:
                print("Error splitting body label file line %s into two parts"%l)
                raise exit(0)
            bodydict[int(lparts[0])] = lparts[1].strip()
        f.close()

    for view in range(2): # 0 for front and 1 for side
        if args.detect:
            tf.reset_default_graph()
        conf = apt.create_conf(lbl_file,view=view,name=name,cache_dir=cache_dir,net_type=model_type)
        update_conf(conf)
        if view ==0:
            # from stephenHeadConfig import sideconf as conf
            extrastr = '_side'
            valmovies = smovies
        else:
            # For FRONT
            # from stephenHeadConfig import conf as conf
            extrastr = '_front'
            valmovies = fmovies

        if args.detect:
            for try_num in range(4):
                try:
                    tf.reset_default_graph()
                    # pred_fn,close_fn,model_file = apt.get_unet_pred_fn(conf)
                    pred_fn, close_fn, model_file = apt.get_pred_fn(model_type=model_type,conf=conf)
                    # self = PoseUNet.PoseUNet(conf, args.net_name)
                    # sess = self.init_net_meta(1)
                    break
                except ValueError:
                    print('Loading the net failed, retrying')
                    if try_num is 3:
                        raise ValueError('Couldnt load the network after 4 tries')

        for ndx in range(len(valmovies)):
            mname,_ = os.path.splitext(os.path.basename(valmovies[ndx]))
            oname = re.sub('!','__',getexpname(valmovies[ndx]))
            pname = os.path.join(args.outdir , oname + extrastr)

            print(oname)

            # detect
            if args.detect and os.path.isfile(valmovies[ndx]) and \
               (args.redo or not os.path.isfile(pname + '.mat')):

                cap = cv2.VideoCapture(valmovies[ndx])
                height = int(cap.get(cvc.FRAME_HEIGHT))
                width = int(cap.get(cvc.FRAME_WIDTH))
                cap.release()
                try:
                    dirname = os.path.normpath(valmovies[ndx])
                    dir_parts = dirname.split(os.sep)
                    aa = re.search('fly_*(\d+)', dir_parts[-3])
                    flynum = int(aa.groups()[0])
                except AttributeError:
                    print('Could not find the fly number from movie name')
                    print('{} isnt in standard format'.format(smovies[ndx]))
                    continue
                crop_loc_all = get_crop_locs(bodydict[flynum],view,height,width) # return x first
                try:
                    predLocs, predScores, pred_ulocs, pred_conf = classify_movie(valmovies[ndx], pred_fn, conf, crop_loc_all)
                    # predList = self.classify_movie(valmovies[ndx], sess, flipud=False)
                except KeyError:
                    continue

#                 orig_crop_loc = [crop_loc_all[i]-1 for i in (2,0)] # y first
#                 # rescale = conf.rescale
#                 # crop_loc = [int(x/rescale) for x in orig_crop_loc]
#                 # end_pad = [int((height-conf.imsz[0])/rescale)-crop_loc[0],int((width-conf.imsz[1])/rescale)-crop_loc[1]]
# #                crop_loc = [old_div(x,4) for x in orig_crop_loc]
# #                end_pad = [old_div(height,4)-crop_loc[0]-old_div(conf.imsz[0],4),old_div(width,4)-crop_loc[1]-old_div(conf.imsz[1],4)]
# #                 pp = [(0,0),(crop_loc[0],end_pad[0]),(crop_loc[1],end_pad[1]),(0,0)]
# #                 predScores = np.pad(predScores,pp,mode='constant',constant_values=-1.)
#
#                 predLocs[:,:,0] += orig_crop_loc[1] # x
#                 predLocs[:,:,1] += orig_crop_loc[0] # y

                hdf5storage.savemat(pname + '.mat',{'locs':predLocs,
                                                    'scores':predScores,
                                                    'expname':valmovies[ndx],
                                                    'crop_loc':crop_loc_all,
                                                    'model_file':model_file,
                                                    'ulocs':pred_ulocs,
                                                    'pred_conf':pred_conf
                                                    },
                                    appendmat=False,truncate_existing=True,gzip_compression_level=0)
                del predScores, predLocs

                print('Detecting:%s'%oname)

            # track
            if args.track and view == 1:

                oname_side = re.sub('!','__',getexpname(smovies[ndx]))
                oname_front = re.sub('!','__',getexpname(fmovies[ndx]))
                pname_side = os.path.join(args.outdir , oname_side + '_side.mat')
                pname_front = os.path.join(args.outdir , oname_front + '_front.mat')
                # 3d trajectories
                basename_front,_ = os.path.splitext(fmovies[ndx])
                basename_side,_ = os.path.splitext(smovies[ndx])
                savefile = basename_side+'_3Dres.mat'
                #savefile = os.path.join(args.outdir , oname_side + '_3Dres.mat')
                trkfile_front = basename_front+'.trk'
                trkfile_side = basename_side+'.trk'

                redo_tracking = args.redo or args.redo_tracking
                if os.path.isfile(savefile) and os.path.isfile(trkfile_front) and \
                   os.path.isfile(trkfile_side) and not redo_tracking:
                    print("%s, %s, and %s exist, skipping tracking"%(savefile,trkfile_front,trkfile_side))
                    continue

                try:
                    dirname = os.path.normpath(smovies[ndx])
                    dir_parts = dirname.split(os.sep)
                    aa = re.search('fly_*(\d+)', dir_parts[-3])
                    flynum = int(aa.groups()[0])
                except AttributeError:
                    print('Could not find the fly number from movie name')
                    print('{} isnt in standard format'.format(smovies[ndx]))
                    continue
                #print "Parsed fly number as %d"%flynum
                kinematfile = os.path.abspath(dltdict[flynum])

                jobid = oname_side

                scriptfile = os.path.join(args.outdir , jobid + '_track.sh')
                logfile = os.path.join(args.outdir , jobid + '_track.log')
                errfile = os.path.join(args.outdir , jobid + '_track.err')


                #print "matscript = " + matscript
                #print "pname_front = " + pname_front
                #print "pname_side = " + pname_side
                #print "kinematfile = " + kinematfile
                
                # make script to be qsubbed
                scriptf = open(scriptfile,'w')
                scriptf.write('if [ -d %s ]\n'%args.outdir)
                scriptf.write('  then export MCR_CACHE_ROOT=%s/mcrcache%s\n'%(args.outdir,jobid))
                scriptf.write('fi\n')
                scriptf.write('%s "%s" "%s" "%s" "%s" "%s" "%s"\n'%(matscript,savefile,pname_front,pname_side,kinematfile,trkfile_front,trkfile_side))
                scriptf.write('chmod g+w {}\n'.format(savefile))
                scriptf.write('chmod g+w {}\n'.format(trkfile_front))
                scriptf.write('chmod g+w {}\n'.format(trkfile_side))
                scriptf.close()
                os.chmod(scriptfile,stat.S_IRUSR|stat.S_IRGRP|stat.S_IWUSR|stat.S_IWGRP|stat.S_IXUSR|stat.S_IXGRP)

#                cmd = "ssh login1 'source /etc/profile; qsub -pe batch %d -N %s -j y -b y -o '%s' -cwd '\"%s\"''"%(args.ncores,jobid,logfile,scriptfile)
                cmd = "ssh 10.36.11.34 'source /etc/profile; bsub -n %d -J %s -oo '%s' -eo '%s' -cwd . '\"%s\"''"%(args.ncores,jobid,logfile,errfile,scriptfile)
                print(cmd)
                call(cmd,shell=True)
                
if __name__ == "__main__":
   main(sys.argv[1:])


def test_crop():
    import trackStephenHead_KB as ts
    import APT_interface as apt
    import multiResData
    import cv2
    from cvc import cvc
    import os
    import re
    import hdf5storage
    crop_reg_file = '/groups/branson/bransonlab/mayank/stephen_copy/crop_regression_params.mat'
    # lbl_file = '/groups/branson/bransonlab/mayank/stephen_copy/apt_cache/sh_trn4523_gtcomplete_cacheddata_bestPrms20180920_retrain20180920T123534_withGTres.lbl'
    lbl_file = '/groups/branson/bransonlab/mayank/stephen_copy/apt_cache/sh_trn4879_gtcomplete.lbl'
    crop_size = [[230, 350], [350, 350]]
    name = 'stephen_20181029'
    cache_dir = '/groups/branson/bransonlab/mayank/stephen_copy/apt_cache'
    bodylblfile = '/groups/branson/bransonlab/mayank/stephen_copy/fly2BodyAxis_lookupTable_Ben.csv'
    import h5py
    bodydict = {}
    f = open(bodylblfile, 'r')
    for l in f:
        lparts = l.split(',')
        if len(lparts) != 2:
            print("Error splitting body label file line %s into two parts" % l)
            raise exit(0)
        bodydict[int(lparts[0])] = lparts[1].strip()
    f.close()

    flynums = [[], []]
    crop_locs = [[], []]
    for view in range(2):
        conf = apt.create_conf(lbl_file, view, 'aa', cache_dir='/groups/branson/home/kabram/temp')
        movs = multiResData.find_local_dirs(conf)[0]

        for mov in movs:
            dirname = os.path.normpath(mov)
            dir_parts = dirname.split(os.sep)
            aa = re.search('fly_*(\d+)', dir_parts[-3])
            flynum = int(aa.groups()[0])
            if bodydict.has_key(flynum):
                cap = cv2.VideoCapture(mov)
                height = int(cap.get(cvc.FRAME_HEIGHT))
                width = int(cap.get(cvc.FRAME_WIDTH))
                cap.release()
                crop_locs[view].append(ts.get_crop_locs(bodydict[flynum], view, height, width))  # return x first
                flynums[view].append(flynum)

    hdf5storage.savemat('/groups/branson/bransonlab/mayank/stephen_copy/auto_crop_locs_trn4879',
                        {'flynum': flynums, 'crop_locs': crop_locs})


def train():
    import PoseUNet_resnet as PoseURes
    import tensorflow as tf

    dstr = PoseTools.datestr()
    cur_name = 'stephen_{}'.format(dstr)

    for view in range(2):
        conf = apt.create_conf(lbl_file,view=view,name=cur_name,cache_dir=cache_dir,net_type=model_type)
        update_conf(conf)
        apt.create_tfrecord(conf, False, use_cache=True)
        tf.reset_default_graph()
        self = PoseURes.PoseUMDN_resnet(conf, name='deepnet')
        self.train_data_name = 'traindata'
        self.train_umdn()
