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

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-only_detect",dest='detect',action="store_true",
                        help="Do only the detection part of tracking which requires GPU")
    group.add_argument("-only_track",dest='track',action="store_true",
                        help="Do only the tracking part of the tracking which requires MATLAB")
    


    args = parser.parse_args()
    if args.redo is None:
        args.redo = False
    if args.redo_tracking is None:
        args.redo_tracking = False
        
    if args.detect is False and args.track is False: 
        args.detect = True
        args.track = True
        
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
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        for view in range(2): # 0 for front and 1 for side
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

            # conf.batch_size = 1

            self = PoseTools.create_network(conf, outtype)
            sess = tf.Session()
            PoseTools.init_network(self, sess, outtype)


            for ndx in range(len(valmovies)):
                mname,_ = os.path.splitext(os.path.basename(valmovies[ndx]))
                oname = re.sub('!','__',conf.getexpname(valmovies[ndx]))
            #     pname = '/groups/branson/home/kabram/bransonlab/PoseTF/results/headResults/movies/' + oname + extrastr
                pname = os.path.join(args.outdir , oname + extrastr)
                print(oname)
                if os.path.isfile(pname + '.mat') and not args.redo:
                    continue


                if not os.path.isfile(valmovies[ndx]):
                    continue

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
                print('Done:%s'%oname)
                
                
    if args.track:

        script_path = os.path.realpath(__file__)
        [script_dir,script_name] = os.path.split(script_path)
        matdir = os.path.join(script_dir,'matlab')
        redo_tracking = args.redo or args.redo_tracking
        matlab_cmd = "addpath %s; GMMTrack2DTo3D('%s','%s','%s','%s',%d);exit;" %(matdir,args.ffilename,
                                                              args.sfilename,args.dltfilename,
                                                              args.outdir,redo_tracking)
        matlab_cmd = 'matlab -nodesktop -nosplash -r "%s" ' % matlab_cmd
        print('Executing matlab command:%s'%matlab_cmd)
        call(matlab_cmd,shell=True)

if __name__ == "__main__":
   main(sys.argv[1:])

