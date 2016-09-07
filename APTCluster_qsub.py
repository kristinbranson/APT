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

    epilogstr = 'Examples for --cmd:\n--cmd "<projname> retrain" # performs full retraining of project\n--cmd "<projname> track <moviename>" # tracks movie using tracker in project\n--cmd "<projname> trackbatch <movielistfile>" # tracks list of movies in text file <movielistfile>'
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,epilog=epilogstr)
    parser.add_argument("--cmd",help="Arguments passed to APTCluster.m")
    parser.add_argument("--multithreaded",action="store_true")
#    parser.add_argument("--elist",help="file containing list of experiments to process")
#    parser.add_argument("--bindir",default="/groups/flyprojects/home/leea30/git/fba.build/bubble/current")
#    parser.add_argument("--setdir",default="/groups/flyprojects/home/leea30/git/fba.flybubble/settings")
    parser.add_argument("--account",default="",help="account to bill for cluster time")
    parser.add_argument("--outdir",default=os.environ['HOME'],help="location to output qsub script and output log")
    parser.add_argument("-pebatch",default="1",help="number of cluster slots to use")
#    parser.add_argument("exps",nargs="*",help="full path to experiments to process")

    args = parser.parse_args()
    
    # misc other args maybe settable in future
    if args.multithreaded:
        args.BIN = "/groups/branson/home/leea30/aptbuild/current/APTCluster/run_APTCluster_multithreaded.sh"
    else:
        args.BIN = "/groups/branson/home/leea30/aptbuild/current/APTCluster/run_APTCluster_singlethreaded.sh"

    args.KEYWORD = "apt"; # used for log/sh filenames, sge job name
    args.MCR = "/groups/branson/home/leea30/mlrt/v90"
    args.USERNAME = subprocess.check_output("whoami").strip()
    args.TMP_ROOT_DIR = "/scratch/" + args.USERNAME
    args.MCR_CACHE_ROOT = args.TMP_ROOT_DIR + "/mcr_cache_root"
    args.QSUBARGS = "-pe batch " + args.pebatch + " -l sl7=true -j y -b y -cwd"    
        
    # summarize for user, proceed y/n?
    argsdisp = vars(args).copy()
    del argsdisp['MCR_CACHE_ROOT']
    del argsdisp['TMP_ROOT_DIR']
    del argsdisp['MCR']
    del argsdisp['KEYWORD']
    
    pprintdict(argsdisp)
    resp = raw_input("Proceed? y/[n]")
    if not resp=="y":
        exit()

    DTPAT = '[0-9]{8,8}T[0-9]{6,6}'

    # jobid
    nowstr = datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")
    nowstr = nowstr[:-3] # keep only milliseconds
    jobid = args.KEYWORD + "-" + nowstr
    print(jobid)

    # generate code
    shfile = os.path.join(args.outdir,"{0:s}.sh".format(jobid))
    logfile = os.path.join(args.outdir,"{0:s}.log".format(jobid))
    gencode(shfile,jobid,args)

    # submit 
    qargs = "-o {0:s} -N {1:s} {2:s} {3:s}".format(logfile,jobid,args.QSUBARGS,shfile)
    if args.account:
        qargs = "-A {0:s} ".format(args.account) + qargs

    qsubcmd = "qsub " + qargs
    print(qsubcmd)
    subprocess.call(qsubcmd,shell=True)

    exit()

def gencode(fname,jobid,args):
    f = open(fname,'w')
    print("#!/bin/bash",file=f)
    print("",file=f)
    print("source ~/.bashrc",file=f)
    print("umask 002",file=f)
    print("export MCR_CACHE_ROOT="+args.MCR_CACHE_ROOT + "." + jobid,file=f)
    print("echo $MCR_CACHE_ROOT",file=f)

    print("",file=f)
    print(args.BIN + " " + args.MCR + " " + args.cmd,file=f)
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

