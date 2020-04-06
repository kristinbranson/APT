#!/usr/bin/env python2

from __future__ import print_function
import sys
import os
import json
import stat
import argparse
import subprocess
import datetime
import re
import glob
import csv
import warnings

def main():

    epilogstr = 'Examples:\n.../runpostproc.py /path/to/rootdir --prmfiles prmfile1.mat#prmfile2.mat --prmpchdir patchdir --outdir outdir\n'

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,epilog=epilogstr)
    parser.add_argument("--rootdir",help="Root directory relative to which other args are specified. Defaults to current dir")
    parser.add_argument("--hmdirfile",help="text file containing hmdirs") # one of hmdirfile or prmpchdir must be specified
    parser.add_argument("--prmfiles",help="base parameter files, separated by '#', relative to rootdir")
    parser.add_argument("--prmpchdir",help="dir containing parameter patches, relative to rootdir") # one of hmdirfile or prmpchdir must be specified
    parser.add_argument("--outdirbase",help="dir where results will be placed, relative to rootdir")
    parser.add_argument("--dryrun",help="if true, print cmds only",action="store_true",default=False)
    parser.add_argument("-nslots",help="num slots")

    args = parser.parse_args()

    if not args.rootdir:
        args.rootdir = os.getcwd()        
    if not os.path.exists(args.rootdir):
        sys.exit("Cannot find rootdir: {}".format(args.rootdir))
        
    tfhmdirfile = bool(args.hmdirfile)
    if tfhmdirfile:
        hmdirfile = os.path.join(args.rootdir,args.hmdirfile)
        with open(hmdirfile) as f:
            hmdirlist = f.readlines()
        hmdirlist = [x.strip() for x in hmdirlist]

    tfpchdir = bool(args.prmpchdir)
    if tfpchdir:
        pchdir = os.path.join(args.rootdir,args.prmpchdir)
        pchglob = os.path.join(pchdir,'*.m')
        pchlist = glob.glob(pchglob)
        pchlist.sort()
        print("Found {} patches in pchdir {}.".format(len(pchlist),args.prmpchdir))

    if tfhmdirfile and tfpchdir:
        sys.exit("Cannot specify both hmdirfile and pchdir.")
    
    nowstr = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    outdirbase = args.outdirbase + "_" + nowstr
    outdir = os.path.join(args.rootdir,outdirbase)
    os.mkdir(outdir)
    print("Made output dir {}.".format(outdir))

    args.bin = os.path.join(args.rootdir,"build","run_RunPostProcessing_HeatmapData2.sh")
    args.mcr = "/misc/local/matlab-2018b/"
    args.TMP_ROOT_DIR = "/scratch/`whoami`"
    args.MCR_CACHE_ROOT = args.TMP_ROOT_DIR + "/mcr_cache_root"    
    
    njobsubmit = 0

    if tfhmdirfile:
        for imov in range(0,len(hmdirlist)):
            hmdir = hmdirlist[imov]
            hmdirS = os.path.basename(hmdir)
            nowstr = datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")
            jobid = "imov{0:04d}_".format(imov) + hmdirS + "_" + nowstr
            shfile = os.path.join(outdir,jobid + ".sh")
            logfile = os.path.join(outdir,jobid + ".log")
            outfile = os.path.join(outdirbase,jobid + ".mat") 

            scriptcmd = [hmdir,
                "rootdir",args.rootdir,
                "paramfiles",args.prmfiles,
                "savefile",outfile]
            scriptcmd = " ".join(scriptcmd)
        
            gencode(shfile,jobid,args,scriptcmd)

            bsubcmd = "bsub -n {} -o {} -J {} {}".format(args.nslots,logfile,jobid,shfile)
            print(bsubcmd)
            if not args.dryrun:
                subprocess.call(bsubcmd,shell=True)
                njobsubmit = njobsubmit+1

        print("{} jobs submitted.".format(njobsubmit))

    elif tfpchdir:
        for pch in pchlist:
            [pchS,pchE] = os.path.splitext(pch)
            pchSNE  = os.path.basename(pchS)
            pchS = os.path.basename(pch)
            
            jobid = pchSNE + nowstr
            shfile = os.path.join(outdir,pchSNE + ".sh")
            logfile = os.path.join(outdir,pchSNE + ".log")
            outfileS = pchSNE + ".mat"
            outfile = os.path.join(outdirbase,outfileS) 
            
            prmfiles = args.prmfiles + "#" + os.path.join(args.prmpchdir,pchS)
            scriptcmd = [
                "DUMMY_HMDIR",
                "rootdir",args.rootdir,
                "paramfiles",prmfiles,
                "savefile",outfile]
            scriptcmd = " ".join(scriptcmd)
        
            gencode(shfile,jobid,args,scriptcmd)

            bsubcmd = "bsub -n {} -o {} -J {} {}".format(args.nslots,logfile,jobid,shfile)
            print(bsubcmd)
            if not args.dryrun:
                subprocess.call(bsubcmd,shell=True)
                njobsubmit = njobsubmit+1

        print("{} jobs submitted.".format(njobsubmit))


def gencode(fname,jobid,args,cmd,bin=None,mcr=None):

    if bin is None:
        bin = args.bin
    if mcr is None:
        mcr = args.mcr

    f = open(fname,'w')
    print("#!/bin/bash",file=f)
    print("",file=f)
    print("source ~/.bashrc",file=f)
    print("umask 002",file=f)
    print("unset DISPLAY",file=f)
    print("if [ -d "+args.TMP_ROOT_DIR+" ]; then",file=f)
    print("  export MCR_CACHE_ROOT="+args.MCR_CACHE_ROOT + "." + jobid,file=f)
    print("fi",file=f)
    print("echo MCR_CACHE_ROOT = $MCR_CACHE_ROOT",file=f)

    print("",file=f)
    print(bin + " " + mcr + " " + cmd,file=f)
    print("",file=f)

    print("if [ -e "+args.MCR_CACHE_ROOT+"."+jobid+" ]; then",file=f)
    print("  echo deleting "+args.MCR_CACHE_ROOT+"."+jobid,file=f)
    print("  date",file=f)
    print("  rm -rf "+args.MCR_CACHE_ROOT+"."+jobid,file=f)
    print("  date",file=f)
    print("fi",file=f)

    f.close()
    os.chmod(fname,stat.S_IRUSR|stat.S_IXUSR|stat.S_IRGRP|stat.S_IXGRP|stat.S_IROTH);


if __name__=="__main__":
    main()
