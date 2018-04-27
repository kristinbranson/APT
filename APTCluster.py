#!/usr/bin/env python2

from __future__ import print_function
import sys
import os
import stat
import argparse
import subprocess
import datetime
import re
import glob
import csv
import warnings
import numpy as np

USEQSUB = False

def main():

    epilogstr = 'Examples:\n.../APTCluster.py /path/to/proj.lbl -n 6 retrain\n.../APTCluster.py /path/to/proj.lbl track -n 4 --mov /path/to/movie.avi\n.../APTCluster.py /path/to/proj.lbl track -n 4 --mov /path/to/movie.avi --trx /path/to/trx.mat\n.../APTCluster.py /path/to/proj.lbl trackbatch -n 2 --movbatchfile /path/to/movielist.txt\n'

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,epilog=epilogstr)

    parser.add_argument("projfile",help="APT project file")
    parser.add_argument("action",choices=["retrain","track","trackbatch","trackbatchserial"],help="action to perform on/with project; one of {retrain, track, trackbatch, trackbatchserial}",metavar="action")

    parser.add_argument("-n","--nslots","--pebatch",help="(required) number of cluster slots",required=True,metavar="NSLOTS")
    parser.add_argument("--mov",help="moviefile; used for action==track",metavar="MOVIE")
    parser.add_argument("--trx",help="trxfile; used for action==track",metavar="TRX")
    parser.add_argument("--movbatchfile",help="file containing list of movies and (optionally) trxfiles; used when action==trackbatch*",metavar="BATCHFILE")
    parser.add_argument("--singlethreaded",help="if true, force run singlethreaded binary",action="store_true",default=False)
    parser.add_argument("--account",default="",help="account to bill for cluster time",metavar="ACCOUNT")
    parser.add_argument("--outdir",help="location to output qsub script and output log. If not supplied, output is written alongside project or movies, depending on action",metavar="PATH")
    parser.add_argument("--bindate",help="APTCluster build date/folder. Defaults to 'current'") 
    parser.add_argument("-l1","--movbatchfilelinestart",help="use with --movbatchfile; start at this line of batchfile (1-based)")
    parser.add_argument("-l2","--movbatchfilelineend",help="use with --movbatchfile; end at this line (inclusive) of batchfile (1-based)")
    parser.add_argument("--trackargs",help="use with action==track or trackbatch. enclose in quotes, additional/optional prop-val pairs")
    parser.add_argument("-p0di","--p0DiagImg",help="use with action==track or trackbatch. short filename for shape initialization diagnostics image")
    parser.add_argument("--mcr",help="mcr to use, eg v90, v901",default="v90")
#    parser.add_argument("--trkfile",help="use with action==prune*. full path to trkfile to prune")
#    parser.add_argument("--pruneargs",help="use with action=prune*. enclose in quotes; '<sigd> <ipt> <frmstart> <frmend>'")
#    parser.add_argument("--prunesig")
    parser.add_argument("-f","--force",help="if true, don't ask for confirmation",action="store_true",default=False)
    parser.add_argument("--splitframes",
                        help="Number of frames to track in each job. If 0, track all frames in one job.",
                        default=0,metavar="SPLITFRAMES",type=int)

    args = parser.parse_args()
    
    if not os.path.exists(args.projfile):
        sys.exit("Cannot find project file: {0:s}".format(args.projfile))

    if args.action=="track":
        if not args.mov:
            sys.exit("--mov must be specified for action==track")
        elif not os.path.exists(args.mov):
            sys.exit("Cannot find movie: {0:s}".format(args.mov))
        if args.trx and not os.path.exists(args.trx):
            sys.exit("Cannot find trx: {0:s}".format(args.trx))
    if args.action in ["trackbatch","trackbatchserial"]:
        if not args.movbatchfile:
            sys.exit("--movbatchfile must be specified for action==trackbatch or trackbatchserial")
        elif not os.path.exists(args.movbatchfile):
            sys.exit("Cannot find movie batchfile: {0:s}".format(args.movbatchfile))
        if args.movbatchfilelinestart:
            args.movbatchfilelinestart = int(args.movbatchfilelinestart)
        else:
            args.movbatchfilelinestart = 1
        if args.movbatchfilelineend:
            args.movbatchfilelineend = int(args.movbatchfilelineend)
        else:
            args.movbatchfilelineend = sys.maxint
    if args.action!="track" and args.mov:
        print("Action is " + args.action + ", ignoring --mov specification")
    if args.action!="track" and args.trx:
        print("Action is " + args.action + ", ignoring --trx specification")
    if args.action not in ["track","trackbatch"] and args.p0DiagImg:
        print("Action is " + args.action + ", ignoring --p0DiagImg specification")    
    if args.action not in ["track","trackbatch"] and args.trackargs:
        print("Action is " + args.action + ", ignoring --trackargs specification")
    if args.action not in ["trackbatch","trackbatchserial"] and args.movbatchfile:
        print("Action is " + args.action + ", ignoring --movbatchfile specification")
        
    args.APTBUILDROOTDIR = "/groups/branson/home/leea30/aptbuild" # root location of binaries
    args.TMPKBBUILDROOTDIR = "/groups/branson/home/bransonk/tracking/code/APT"
    args.TMPKBMCR = "/groups/branson/bransonlab/projects/olympiad/MCR/v92"
    if not args.bindate:
        args.bindate = "current"
    args.binroot = os.path.join(args.APTBUILDROOTDIR,args.bindate)

    args.multithreaded = not args.singlethreaded and int(args.nslots)>1
    if args.multithreaded:
        args.bin = os.path.join(args.binroot,"APTCluster","run_APTCluster_multithreaded.sh")
    else:
        args.bin = os.path.join(args.binroot,"APTCluster","run_APTCluster_singlethreaded.sh")
    if not os.path.exists(args.bin):
        sys.exit("Cannot find binary: {0:s}".format(args.bin))

    # KB: binary for getting info about movies
    # todo - get Allen to put this in his in his build directory
    args.infobin = os.path.join(args.TMPKBBUILDROOTDIR,"GetMovieNFrames","for_redistribution_files_only","run_GetMovieNFrames.sh")

    # check for mlrt tokens to specify/override mcr
    bindir = os.path.dirname(args.bin)
    mlrtTok = glob.glob(os.path.join(bindir,"MLRT_*"))
    if len(mlrtTok)>1:
        warnings.warn("More than one MLRT_ token found in bindir: {0:s}".format(bindir))
    if mlrtTok:
        mlrtTok = os.path.basename(mlrtTok[-1])
        mlrtMcr = mlrtTok[5:]
        print("Found token in bindir: {0:s}. Using --mcr: {1:s}".format(mlrtTok,mlrtMcr))
        args.mcr = mlrtMcr

    args.KEYWORD = "apt" # used for log/sh filenames, sge job name
    args.MCRROOT = "/groups/branson/home/leea30/mlrt/"
    args.MCR = os.path.join(args.MCRROOT,args.mcr)
    if not os.path.exists(args.MCR):
        sys.exit("Cannot find mcr: {0:s}".format(args.MCR))
    #args.USERNAME = subprocess.check_output("whoami").strip()
    args.TMP_ROOT_DIR = "/scratch/`whoami`"
    args.MCR_CACHE_ROOT = args.TMP_ROOT_DIR + "/mcr_cache_root"

    if USEQSUB:
        args.BSUBARGS = "-pe batch " + args.nslots + " -j y -b y -cwd" 
        if args.account:
            args.BSUBARGS = "-A {0:s} ".format(args.account) + args.BSUBARGS
    else:
        args.BSUBARGS = "-n " + args.nslots 
        if args.account:
            args.BSUBARGS = "-P {0:s} ".format(args.account) + args.BSUBARGS
        
    # summarize for user, proceed y/n?
    argsdisp = vars(args).copy()
    argsdispRmFlds = ['MCR_CACHE_ROOT','TMP_ROOT_DIR','MCR','KEYWORD','bindate','binroot','nslots','account','multithreaded']
    for fld in argsdispRmFlds:
        del argsdisp[fld]    
    if not args.force:
        pprintdict(argsdisp)
        resp = raw_input("Proceed? y/[n]")
        if not resp=="y":
            sys.exit("Aborted")

    if args.outdir and not os.path.exists(args.outdir):
        print("Creating outdir: " + args.outdir)
        os.makedirs(args.outdir)

    if args.action=="trackbatch":
        with open(args.movbatchfile,'r') as csvfile:
            csvr = csv.reader(csvfile,delimiter=',')
            movlist = list(csvr) # list of lists

        imov0 = args.movbatchfilelinestart-1
        imov1 = args.movbatchfilelineend # one past end
        movlist = movlist[imov0:imov1]

        nmovtot = len(movlist)
        nmovsub = 0
        for lst in movlist:
            if len(lst)==1:
                mov = lst[0]
                trx = "''"
            elif len(lst)==2:
                mov = lst[0]
                trx = lst[1]
            else:
                print("Skipping invalid line encountered in movielist...")
                continue

            mov = mov.rstrip() # prob unnec
            trx = trx.rstrip() # etc
            if not os.path.exists(mov):
                print("Cannot find movie: " + mov + ". Skipping...")
                continue
            if trx!="''" and not os.path.exists(trx):
                print("Cannot find trx: " + trx + ". Skipping...")
                continue
            
            # jobid
            nowstr = datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")
            nowstr = nowstr[:-3] # keep only milliseconds
            jobid = args.KEYWORD + "-" + nowstr
            print(jobid)

            # generate code
            if args.outdir:
                outdiruse = args.outdir
            else:
                outdiruse = os.path.dirname(mov)
            shfile = os.path.join(outdiruse,"{0:s}.sh".format(jobid))
            logfile = os.path.join(outdiruse,"{0:s}.log".format(jobid))

            cmd = args.projfile + " track  " + mov + " " + trx
            if args.trackargs:
                cmd = cmd + " " + args.trackargs
            if args.p0DiagImg:
                p0DiagImgFull = os.path.join(outdiruse,args.p0DiagImg) # won't work well when args.outdir supplied
                cmd = cmd + " p0DiagImg " + p0DiagImgFull
            gencode(shfile,jobid,args,cmd)

            # submit 
            if USEQSUB:
                qargs = "-o {0:s} -N {1:s} {2:s} {3:s}".format(logfile,jobid,args.BSUBARGS,shfile)
                qsubcmd = "qsub " + qargs
            else:
                qargs = '{0:s} -R"affinity[core(1)]" -o {1:s} -J {2:s} {3:s}'.format(args.BSUBARGS,logfile,jobid,shfile)
                qsubcmd = "bsub " + qargs
            print(qsubcmd)
            subprocess.call(qsubcmd,shell=True)
            nmovsub = nmovsub+1

    else:
        # jobid
        nowstr = datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")
        nowstr = nowstr[:-3] # keep only milliseconds
        jobid = args.KEYWORD + "-" + nowstr
        print(jobid)

        # generate code
        if args.outdir:
            outdiruse = args.outdir
        else:
            if args.action=="track":
                outdiruse = os.path.dirname(args.mov)
            else: # trackbatchserial, retrain
                outdiruse = os.path.dirname(args.projfile)                
        shfile = os.path.join(outdiruse,"{0:s}.sh".format(jobid))
        logfile = os.path.join(outdiruse,"{0:s}.log".format(jobid))
        if args.action=="retrain":
            cmd = args.projfile + " " + args.action
        elif args.action=="track":

            if args.trx:
                cmd=args.projfile+"  "+args.action+" "+args.mov+" "+args.trx
            else:
                cmd=args.projfile+"  "+args.action+" "+args.mov+" "+"''"
            if args.trackargs:
                cmd=cmd+" "+args.trackargs
            if args.p0DiagImg:
                p0DiagImgFull=os.path.join(outdiruse,args.p0DiagImg)
                cmd=cmd+" p0DiagImg "+p0DiagImgFull

            if args.splitframes > 0:
                infocmd = [args.infobin,args.TMPKBMCR,args.mov]
                s = subprocess.check_output(infocmd)
                p=re.compile('\n\d+$') # last number
                m = p.search(s)
                s = s[m.start()+1:-1]
                nframes = int(s)
                njobs = np.maximum(1,np.round(nframes/args.splitframes))
                jobstarts = np.round(np.linspace(1,nframes+1,njobs+1)).astype(int)
                jobends = jobstarts[1:]-1

                jobinfofile=os.path.join(outdiruse,"splittrackinfo_{0:s}.txt".format(jobid))
                f=open(jobinfofile,'w')

                for jobi in range(njobs):
                    jobidcurr = "%s-%03d"%(jobid,jobi)
                    if args.outdir:
                        rawtrkname='%s/$movfile_$projfile_%s'%(outdiruse,jobidcurr)
                    else:
                        rawtrkname = '$movdir/$movfile_$projfile_%s'%(jobidcurr)
                    cmdcurr = "%s startFrame %d endFrame %d rawtrkname %s"%(cmd,jobstarts[jobi],jobends[jobi],rawtrkname)
                    shfilecurr = os.path.join(outdiruse,"{0:s}.sh".format(jobidcurr))
                    logfilecurr = os.path.join(outdiruse,"{0:s}.log".format(jobidcurr))

                    infoline = "%d,%d,%d,%s,%s,%s,%s"%(jobi,jobstarts[jobi],jobends[jobi],jobidcurr,rawtrkname,shfilecurr,logfilecurr)
                    f.write(infoline)
                    gencode(shfilecurr,jobidcurr,args,cmdcurr)

                    # submit
                    if USEQSUB:
                        qargs = "-o {0:s} -N {1:s} {2:s} {3:s}".format(logfilecurr,jobidcurr,args.BSUBARGS,shfilecurr)
                        qsubcmd = "qsub " + qargs
                    else:
                        qargs = '{0:s} -R"affinity[core(1)]" -o {1:s} -J {2:s} {3:s}'.format(args.BSUBARGS,logfilecurr,jobidcurr,shfilecurr)
                        qsubcmd = "bsub " + qargs


                    print(qsubcmd)
                    #subprocess.call(qsubcmd,shell=True)

                f.close()

                print("%d jobs submitted, information about them in file %s."%(njobs,jobinfofile))

                sys.exit()

        elif args.action=="trackbatchserial":
            cmd = args.projfile + "  trackbatch " + args.movbatchfile

        gencode(shfile,jobid,args,cmd)

        # submit 
        if USEQSUB:
            qargs = "-o {0:s} -N {1:s} {2:s} {3:s}".format(logfile,jobid,args.BSUBARGS,shfile)
            qsubcmd = "qsub " + qargs
        else:
            qargs = '{0:s} -R"affinity[core(1)]" -o {1:s} -J {2:s} {3:s}'.format(args.BSUBARGS,logfile,jobid,shfile)
            qsubcmd = "bsub " + qargs

        print(qsubcmd)
        subprocess.call(qsubcmd,shell=True)

    sys.exit()

def gencode(fname,jobid,args,cmd,bin=None,MCR=None):

    if bin is None:
        bin = args.bin
    if MCR is None:
        MCR = args.MCR

    f = open(fname,'w')
    print("#!/bin/bash",file=f)
    print("",file=f)
    print("source ~/.bashrc",file=f)
    print("umask 002",file=f)
    print("unset DISPLAY",file=f)
    print("if [ -d "+args.MCR_CACHE_ROOT+" ]; then",file=f)
    print("  export MCR_CACHE_ROOT="+args.MCR_CACHE_ROOT + "." + jobid,file=f)
    print("fi",file=f)
    print("echo $MCR_CACHE_ROOT",file=f)

    print("",file=f)
    print(bin + " " + MCR + " " + cmd,file=f)
    print("",file=f)

    print("rm -rf",args.MCR_CACHE_ROOT+"."+jobid,file=f)
    f.close()
    os.chmod(fname,stat.S_IRUSR|stat.S_IXUSR|stat.S_IRGRP|stat.S_IXGRP|stat.S_IROTH);

def pprintdict(d, indent=0):
   for key, value in sorted(d.items()):
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pprintdict(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

if __name__=="__main__":
    print("main")
    main()

