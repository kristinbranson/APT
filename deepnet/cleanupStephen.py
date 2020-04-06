# coding: utf-8

# In[13]:

# !/usr/bin/python
import os
import re
import sys
import argparse
import stat
import time
import shutil

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("-o",dest='outdir',
                        help="temp out directory",
                        required=True)
    parser.add_argument("-s",dest="sfilename",
                      help="text file with list of side view videos",
                      required=True)
    parser.add_argument("-f",dest="ffilename",
                      help="text file with list of front view videos. The list of side view videos and front view videos should match up",
                      required=True)
    parser.add_argument("-d",dest="days",type=int,
                      help="delete temp files only if the trk file has been created within d days",
                      default=3)

    args = parser.parse_args()
    args.outdir = os.path.abspath(args.outdir)

    with open(args.sfilename, "r") as text_file:
        smovies = text_file.readlines()
    smovies = [x.rstrip() for x in smovies]
    with open(args.ffilename, "r") as text_file:
        fmovies = text_file.readlines()
    fmovies = [x.rstrip() for x in fmovies]

    current_time = time.time()

    for ndx in range(len(smovies)):
        # 3d trajectories
        basename_front, _ = os.path.splitext(fmovies[ndx])
        basename_side, _ = os.path.splitext(smovies[ndx])
        trkfile_front = basename_front + '.trk'
        trkfile_side = basename_side + '.trk'

        if os.path.isfile(trkfile_front) and os.path.isfile(trkfile_side): 
            front_sec = current_time - os.path.getctime(trkfile_front)
            front_new  =  (front_sec) / (24 * 3600) <= args.days
            side_sec = current_time - os.path.getctime(trkfile_side)
            side_new  =  (side_sec) / (24 * 3600) <= args.days
            if front_new and side_new:

                print('Deleting temp files for {}'.format(smovies[ndx]))
    
                for view in range(2):  # 0 for front and 1 for side
                    if view == 1:
                        from stephenHeadConfig import sideconf as conf
                        extrastr = '_side'
                        cur_movie = smovies[ndx]
                    else:
                        # For FRONT
                        from stephenHeadConfig import conf as conf
                        extrastr = '_front'
                        cur_movie = fmovies[ndx]
    
                    mname, _ = os.path.splitext(os.path.basename(cur_movie))
                    oname = re.sub('!', '__', conf.getexpname(cur_movie))
                    pname = os.path.join(args.outdir, oname + extrastr)
    
                    if os.path.isfile(pname+'.mat'):
                        print('Deleting '+ pname+'.mat')
                        os.unlink(pname+'.mat')
    
                    oname_side = re.sub('!', '__', conf.getexpname(smovies[ndx]))
    
                    jobid = oname_side
    
                    scriptfile = os.path.join(args.outdir, jobid + '_track.sh')
                    logfile = os.path.join(args.outdir, jobid + '_track.log')
                    errfile = os.path.join(args.outdir, jobid + '_track.err')
                    mcrdir = os.path.join(args.outdir, 'mcrcache'+jobid)
                    os.unlink(scriptfile) if os.path.isfile(scriptfile) else None
                    os.unlink(logfile) if os.path.isfile(logfile) else None
                    os.unlink(errfile) if os.path.isfile(errfile) else None
                    shutil.rmtree(mcrdir) if os.path.isdir(mcrdir) else None
    

if __name__ == "__main__":
   main(sys.argv[1:])

