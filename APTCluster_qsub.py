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

    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd",help="APTCluster args")
    parser.add_argument("--multithreaded",action="store_true")
#    parser.add_argument("--elist",help="file containing list of experiments to process")
#    parser.add_argument("--bindir",default="/groups/flyprojects/home/leea30/git/fba.build/bubble/current")
#    parser.add_argument("--setdir",default="/groups/flyprojects/home/leea30/git/fba.flybubble/settings")
#    parser.add_argument("-ap","--anlsprot",default="current_bubble")
    parser.add_argument("--account",default="bransonk",help="account to charge")
    parser.add_argument("--outdir",default="/groups/branson/home/leea30/aptrun",help="output dir")
#    parser.add_argument("--outsubdirauto",action="store_true")
#    parser.add_argument("--aci",action="store_true")
#    parser.add_argument("--reg",action="store_true")
#    parser.add_argument("--sex",action="store_true")
#    parser.add_argument("--wgt",action="store_true")
#    parser.add_argument("--pff",action="store_true")
#    parser.add_argument("--dec",action="store_true")
#    parser.add_argument("--jdt",action="store_true")
#    parser.add_argument("--mov",action="store_true")
    parser.add_argument("-pebatch",default="1",help="number of slots")
#    parser.add_argument("exps",nargs="*",help="full path to experiments to process")

    args = parser.parse_args()
    
    # misc other args maybe settable in future
    if args.multithreaded:
        args.BIN = "/groups/branson/home/leea30/git/apt/APTCluster/multithreaded/run_APTCluster.sh"
    else:
        args.BIN = "/groups/branson/home/leea30/git/apt/APTCluster/singlethreaded/run_APTCluster.sh"

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
    qargs = "-A {0:s} -o {1:s} -N {2:s} {3:s} {4:s}".format(args.account,
                                                            logfile,jobid,args.QSUBARGS,shfile)
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

