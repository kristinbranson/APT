from __future__ import division
from __future__ import print_function

# coding: utf-8

# In[13]:

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

def main(argv):
    print(argv)

    defaulttrackerpath = "/groups/branson/bransonlab/mayank/PoseTF/matlab/compiled/run_compute3Dfrom2D_compiled.sh"
    defaultmcrpath = "/groups/branson/bransonlab/mayank/MCR/v92"

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
    parser.add_argument("-fold",dest="fold",
                      help="fold number",
                      required=True)
    parser.add_argument("-o",dest="outdir",
                      help="temporary output directory to store intermediate computations",
                      required=True)
    parser.add_argument("-r",dest="redo",
                      help="if specified will recompute everything",
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
                        help="Number of cores to assign to each MATLAB tracker job",
                        default=1)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-only_detect",dest='detect',action="store_true",
                        help="Do only the detection part of tracking which requires GPU")
    group.add_argument("-only_track",dest='track',action="store_true",
                        help="Do only the tracking part of the tracking which requires MATLAB")
    


    args = parser.parse_args(argv)
    if args.redo is None:
        args.redo = False
        
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

    if args.track:
        if len(smovies) != len(fmovies):
            print("Side and front movies must match")
            raise exit(0)

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

        for ff in smovies+fmovies:
            if not os.path.isfile(ff):
                print("Movie %s not found"%(ff))
                raise exit(0)
        if args.gpunum is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpunum)

    for view in range(2): # 0 for front and 1 for side
        if args.detect:
            tf.reset_default_graph() 
        if view ==1:
        
            from stephenHeadConfig import sideconf as conf
            conf.useMRF = True
            outtype = 2
            extrastr = '_side'
            valmovies = smovies

        else:
            # For FRONT
            from stephenHeadConfig import conf as conf
            conf.useMRF = True
            outtype = 2
            extrastr = '_front'
            valmovies = fmovies

        ext = '_fold_{}'.format(args.fold)
        conf.cachedir = os.path.join(conf.cachedir, 'cross_val_fly')
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


        # conf.batch_size = 1

        if args.detect:        
            self = PoseTools.create_network(conf, outtype)
            sess = tf.Session()
            PoseTools.init_network(self, sess, outtype)

        for ndx in range(len(valmovies)):
            mname,_ = os.path.splitext(os.path.basename(valmovies[ndx]))
            oname = re.sub('!','__',conf.getexpname(valmovies[ndx]))
            pname = os.path.join(args.outdir , oname + extrastr)

            print(oname)

            flynum = conf.getflynum(smovies[ndx])
            # print "Parsed fly number as %d"%flynum
            if flynum not in dltdict:
                print('No dlt file, skipping')
                continue

            # detect
            if args.detect and os.path.isfile(valmovies[ndx]) and \
               (args.redo or not os.path.isfile(pname + '.mat')):

                predList = PoseTools.classify_movie(conf, valmovies[ndx], outtype, self, sess)

                if args.makemovie:
                    PoseTools.create_pred_movie(conf, predList, valmovies[ndx], pname + '.avi', outtype)

                cap = cv2.VideoCapture(valmovies[ndx])
                height = int(cap.get(cvc.FRAME_HEIGHT))
                width = int(cap.get(cvc.FRAME_WIDTH))
                orig_crop_loc = conf.cropLoc[(height,width)]
                crop_loc = [old_div(x,4) for x in orig_crop_loc] 
                end_pad = [old_div(height,4)-crop_loc[0]-old_div(conf.imsz[0],4),old_div(width,4)-crop_loc[1]-old_div(conf.imsz[1],4)]
                pp = [(0,0),(crop_loc[0],end_pad[0]),(crop_loc[1],end_pad[1]),(0,0),(0,0)]
                predScores = np.pad(predList[1],pp,mode='constant',constant_values=-1.)

                predLocs = predList[0]
                predLocs[:,:,:,0] += orig_crop_loc[1]
                predLocs[:,:,:,1] += orig_crop_loc[0]

                io.savemat(pname + '.mat',{'locs':predLocs,'scores':predScores[...,0],'expname':valmovies[ndx]})
                print('Detecting:%s'%oname)

            # track
            if args.track and view == 1:

                oname_side = re.sub('!','__',conf.getexpname(smovies[ndx]))
                oname_front = re.sub('!','__',conf.getexpname(fmovies[ndx]))
                pname_side = os.path.join(args.outdir , oname_side + '_side.mat')
                pname_front = os.path.join(args.outdir , oname_front + '_front.mat')
                # 3d trajectories
                basename_front,_ = os.path.splitext(fmovies[ndx])
                basename_side,_ = os.path.splitext(smovies[ndx])
                savefile = basename_side+'_3Dres.mat'
                #savefile = os.path.join(args.outdir , oname_side + '_3Dres.mat')
                trkfile_front = basename_front+'.trk'
                trkfile_side = basename_side+'.trk'

                if os.path.isfile(savefile) and os.path.isfile(trkfile_front) and \
                   os.path.isfile(trkfile_side) and not args.redo:
                    print("%s, %s, and %s exist, skipping tracking"%(savefile,trkfile_front,trkfile_side))
                    continue

                flynum = conf.getflynum(smovies[ndx])
                #print "Parsed fly number as %d"%flynum
                if flynum not in dltdict:
                    continue
                kinematfile = os.path.abspath(dltdict[flynum])

                jobid = oname_side

                scriptfile = os.path.join(args.outdir , jobid + '_track.sh')
                logfile = os.path.join(args.outdir , jobid + '_track.log')


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
                scriptf.close()
                os.chmod(scriptfile,stat.S_IRUSR|stat.S_IRGRP|stat.S_IWUSR|stat.S_IWGRP|stat.S_IXUSR|stat.S_IXGRP)

                cmd = "ssh login1 'source /etc/profile; qsub -pe batch %d -N %s -j y -b y -o '%s' -cwd '\"%s\"''"%(args.ncores,jobid,logfile,scriptfile)
                print(cmd)
                call(cmd,shell=True)
                
if __name__ == "__main__":
   main(sys.argv[1:])

