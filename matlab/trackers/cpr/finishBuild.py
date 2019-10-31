#!/usr/local/anaconda/bin/python

import sys
import os
import shutil
import argparse
import subprocess

PRJS = ['trainAL','testAL','CPRLabelTrackerTrack']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--builddir",default="/groups/branson/home/leea30/git/cpr.build",help="root of cpr build directory")
    args = parser.parse_args()
        
    if not os.path.exists(args.builddir):
        sys.exit('ERROR: Dir %s not found' % args.builddir)

    # get the current HEAD sha
    thisfile = os.path.abspath(__file__)
    cprroot = os.path.dirname(thisfile)
    cmd = "git --git-dir=" + os.path.join(cprroot,".git") + " rev-parse HEAD"
    print(cmd)
    sha = subprocess.check_output(cmd,shell=True)
    sha = str.strip(sha)
    print "the sha of HEAD is " + sha

    shadir = os.path.join(args.builddir,sha)
    if os.path.exists(shadir):
        sys.exit('ERROR: SHA dir %s already exists.' % shadir)
    else:
        os.mkdir(shadir)
        print "made shadir: " + shadir

    currentptr = os.path.join(args.builddir,'current')
    currentptrlink = os.readlink(currentptr)
    print "the current ptr: " + currentptr + " points to: " + currentptrlink

    for prj in PRJS:
        prjfull = os.path.join(cprroot,"misc",prj)
        if os.path.exists(prjfull):            
            print "Found: " + prjfull + ". Moving to shadir."
            shutil.move(prjfull,shadir)
        else:
            linktgt = os.path.join("..",currentptrlink,prj)
            print "NOT found: " + prjfull + ". Linking to " + linktgt
            cmd = "ln -s " + linktgt + " " + shadir
            print(cmd)
            subprocess.call(cmd,shell=True)

    if os.path.exists(currentptr):
        os.remove(currentptr)
    cmd = "ln -s " + sha + " " + currentptr
    print(cmd)
    subprocess.call(cmd,shell=True)
            
#        
#
#    dirNewFull = os.path.join(args.dirBase,args.dirNew)
#    dirTgtFull = os.path.join(args.dirBase,args.dirTgt)
#    if not os.path.exists(dirNewFull):
#        sys.exit('ERROR: Dir %s not found' % dirNewFull)
#    if not os.path.exists(dirTgtFull):
#        sys.exit('ERROR: Dir %s not found' % dirTgtFull)
#
#    for stage in STAGES:
#        binfile = "FlyBubble" + stage
#        runfile = "run_FlyBubble" + stage + ".sh"
#        linkIfDNE(args.dirBase,args.dirNew,args.dirTgt,binfile)
#        linkIfDNE(args.dirBase,args.dirNew,args.dirTgt,runfile)
#        
#
if __name__=="__main__":
    main()
