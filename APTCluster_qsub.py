#!/usr/local/anaconda/bin/python

from __future__ import print_function
import sys
import os
import stat
import argparse
import subprocess
import datetime
import re

def main():

    epilogstr = 'Examples:\n.../APTCluster_qsub.py /path/to/proj.lbl --pebatch 10 --multithreaded retrain\n.../APTCluster_qsub.py /path/to/proj.lbl track --pebatch 8 --mov /path/to/movie.avi\n.../APTCluster_qsub.py /path/to/proj.lbl trackbatch --pebatch 10 --multithreaded --movbatchfile /path/to/movielist.txt\n'
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,epilog=epilogstr)

    parser.add_argument("projfile",help="APT project file")
    parser.add_argument("action",choices=["retrain","track","trackbatch","trackbatchserial"],help="action to perform on/with project; one of {retrain, track, trackbatch, trackbatchserial}",metavar="action")

    parser.add_argument("--pebatch",help="(required) number of cluster slots",required=True,metavar="NSLOTS")
    parser.add_argument("--mov",help="moviefile; used for action==track",metavar="MOVIE")
    parser.add_argument("--movbatchfile",help="file containing list of movies; used when action==trackbatch*",metavar="BATCHFILE")
    parser.add_argument("--multithreaded",help="if true, run multithreaded binary",action="store_true",default=False)
    parser.add_argument("--account",default="",help="account to bill for cluster time",metavar="ACCOUNT")
    parser.add_argument("--outdir",help="location to output qsub script and output log. If not supplied, output is written alongside project or movies, depending on action",metavar="PATH")
    parser.add_argument("--bindate",help="APTCluster build date/folder. Defaults to 'current'") 
    parser.add_argument("-l1","--movbatchfilelinestart",help="use with --movbatchfile; start at this line of batchfile (1-based)")
    parser.add_argument("-l2","--movbatchfilelineend",help="use with --movbatchfile; end at this line (inclusive) of batchfile (1-based)")
    parser.add_argument("--trackargs",help="use with action==track. enclose in quotes, additional/optional prop-val pairs (eg p0DiagImg, trkFilename, stripTrkPFull)")

    args = parser.parse_args()
    
    if not os.path.exists(args.projfile):
        sys.exit("Cannot find project file: {0:s}".format(args.projfile))

    if args.action=="track":
        if not args.mov:
            sys.exit("--mov must be specified for action==track")
        elif not os.path.exists(args.mov):
            sys.exit("Cannot find movie: {0:s}".format(args.mov))
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
    if args.action!="track" and args.trackargs:
        print("Action is " + args.action + ", ignoring --trackargs specification")
    if args.action not in ["trackbatch","trackbatchserial"] and args.movbatchfile:
        print("Action is " + args.action + ", ignoring --movbatchfile specification")
        
    args.APTBUILDROOTDIR = "/groups/branson/home/leea30/aptbuild"
    if not args.bindate:
        args.bindate = "current"
    args.binroot = os.path.join(args.APTBUILDROOTDIR,args.bindate)

    if args.multithreaded:
        args.bin = os.path.join(args.binroot,"APTCluster","run_APTCluster_multithreaded.sh")
    else:
        args.bin = os.path.join(args.binroot,"APTCluster","run_APTCluster_singlethreaded.sh")
    if not os.path.exists(args.bin):
        sys.exit("Cannot find binary: {0:s}".format(args.bin))

    args.KEYWORD = "apt" # used for log/sh filenames, sge job name
    args.MCR = "/groups/branson/home/leea30/mlrt/v90"
    args.USERNAME = subprocess.check_output("whoami").strip()
    args.TMP_ROOT_DIR = "/scratch/" + args.USERNAME
    args.MCR_CACHE_ROOT = args.TMP_ROOT_DIR + "/mcr_cache_root"
    args.QSUBARGS = "-pe batch " + args.pebatch + " -l sl7=true -j y -b y -cwd" 
    if args.account:
        args.QSUBARGS = "-A {0:s} ".format(args.account) + args.QSUBARGS
        
    # summarize for user, proceed y/n?
    argsdisp = vars(args).copy()
    del argsdisp['MCR_CACHE_ROOT']
    del argsdisp['TMP_ROOT_DIR']
    del argsdisp['MCR']
    del argsdisp['KEYWORD']
    
    pprintdict(argsdisp)
    resp = raw_input("Proceed? y/[n]")
    if not resp=="y":
        sys.exit("Aborted")

    if args.outdir and not os.path.exists(args.outdir):
        print("Creating outdir: " + args.outdir)
        os.makedirs(args.outdir)

    if args.action=="trackbatch":
        movs = [line.rstrip('\n') for line in open(args.movbatchfile,'r')]
        imov0 = args.movbatchfilelinestart-1
        imov1 = args.movbatchfilelineend # one past end
        movs = movs[imov0:imov1]

        nmovtot = len(movs)
        nmovsub = 0
        for mov in movs:
            mov = mov.rstrip()

            if not os.path.exists(mov):
                print("Cannot find movie: " + mov + ". Skipping...")
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
            cmd = args.projfile + " track " + mov
            gencode(shfile,jobid,args,cmd)

            # submit 
            qargs = "-o {0:s} -N {1:s} {2:s} {3:s}".format(logfile,jobid,args.QSUBARGS,shfile)
            qsubcmd = "qsub " + qargs
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
            cmd = args.projfile + "  " + args.action + " " + args.mov         
            if args.trackargs:
                 cmd = cmd + args.trackargs
         
        elif args.action=="trackbatchserial":
            cmd = args.projfile + "  trackbatch " + args.movbatchfile

        gencode(shfile,jobid,args,cmd)

        # submit 
        qargs = "-o {0:s} -N {1:s} {2:s} {3:s}".format(logfile,jobid,args.QSUBARGS,shfile)
        qsubcmd = "qsub " + qargs
        print(qsubcmd)
        subprocess.call(qsubcmd,shell=True)

    exit()

def gencode(fname,jobid,args,cmd):
    f = open(fname,'w')
    print("#!/bin/bash",file=f)
    print("",file=f)
    print("source ~/.bashrc",file=f)
    print("umask 002",file=f)
    print("export MCR_CACHE_ROOT="+args.MCR_CACHE_ROOT + "." + jobid,file=f)
    print("echo $MCR_CACHE_ROOT",file=f)

    print("",file=f)
    print(args.bin + " " + args.MCR + " " + cmd,file=f)
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
    main()

