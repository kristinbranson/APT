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


def findRoundIdx(jdat):    
    # determine round/prm/pch
    #
    # Returns roundIdx, prmfull, prmS, pchfull, pchS

    prmglob = os.path.join(jdat['hpo_base_dir'],'prm*.mat')
    pchglob = os.path.join(jdat['hpo_base_dir'],'pch*')
    prmlist = glob.glob(prmglob)
    prmlist.sort()
    prmfull = prmlist[-1]
    prmS = os.path.basename(prmfull)
    pchlist = glob.glob(pchglob)
    pchlist.sort()
    pchfull = pchlist[-1]
    pchS = os.path.basename(pchfull)

    prmpat = 'prm(?P<idx>[0-9]+).mat'
    pchpat = 'pch(?P<idx>[0-9]+)'
    m = re.search(prmpat,prmS)
    if not m:
        sys.exit("Cannot find any parameters with pattern: {0:s}".format(prmpat))
    prmidx = int(m.group('idx'))
    m = re.search(pchpat,pchS)
    if not m:
        sys.exit("Cannot find any pchdirs with pattern: {0:s}".format(pchpat))
    pchidx = int(m.group('idx'))
    if prmidx!=pchidx:
        sys.exit("Latest pch and prms do not match.")

    return prmidx, prmfull, prmS, pchfull, pchS

def findPrmPch(jdat,roundIdx):    
    # prmfull, prmS, pchfull, pchS

    basedir = jdat['hpo_base_dir']
    prm1 = os.path.join(basedir,'prm{0:d}.mat'.format(roundIdx))
    prm2 = os.path.join(basedir,'prm{0:02d}.mat'.format(roundIdx))
    pch1 = os.path.join(basedir,'pch{0:d}'.format(roundIdx))
    pch2 = os.path.join(basedir,'pch{0:02d}'.format(roundIdx))

    if os.path.exists(prm1):
        prmfull = prm1
    elif os.path.exists(prm2):
        prmfull = prm2
    else:
        sys.exit("Cannot find parameter file for round {0:d}".format(roundIdx))

    if os.path.exists(pch1):
        pchfull = pch1
    elif os.path.exists(pch2):
        pchfull = pch2
    else:
        sys.exit("Cannot find pch dir for round {0:d}".format(roundIdx))

    prmS = os.path.basename(prmfull)
    pchS = os.path.basename(pchfull)

    return prmfull, prmS, pchfull, pchS

def main():

    epilogstr = 'Examples:\n.../hpo.py /path/to/hpo_manifest.json\n'

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,epilog=epilogstr)
    parser.add_argument("hpojson",help="HPO manifest json")
    parser.add_argument("action",choices=["xv","xvpch","trntrk"],help="",metavar="action")
    parser.add_argument("--roundidx",help="round idx (usually 0-based) for which trntrk will be run")
    parser.add_argument("--dryrun",help="Show but do not execute cluster (bsub) commands. Code generation still occurs.",action="store_true",default=False)
    args = parser.parse_args()
    
    if not os.path.exists(args.hpojson):
        sys.exit("Cannot find json file: {0:s}".format(args.hpojson))

    with open(args.hpojson) as f:
        jdat = json.load(f)
        
    if args.action=="xv" or args.action=="xvpch": 
        if not args.roundidx:
            roundIdx, prmfull, prmS, pchfull, pchS = findRoundIdx(jdat)
            print("Found latest prm/pchs => roundIdx: {0:d}".format(roundIdx))
        else:
            roundIdx = int(args.roundidx)
            prmfull, prmS, pchfull, pchS = findPrmPch(jdat,roundIdx)
            print("Found prm/pch for roundIdx {0:d}: {1:s}, {2:s}".format(
                roundIdx,prmfull,pchfull))

        # create outputdirs
        for split in jdat['splits']:
            splitifo = jdat['splits'][split]
            splitdir = os.path.join(jdat['hpo_base_dir'],splitifo['dir'])
            if not os.path.exists(splitdir):
                os.mkdir(splitdir)
                print("Split {0:s}: created output dir: {1:s}".format(split,splitdir))
            
            rnddirS = os.path.splitext(prmS)[0]
            rnddir = os.path.join(splitdir,rnddirS)
            if os.path.exists(rnddir):
                if os.listdir(rnddir):
                    sys.exit("Split {0:s}: output/round dir exists and is nonempty: {1:s}".format(split,rnddir))
                else:
                    print("Split {0:s}: found empty output/round dir: {1:s}".format(split,rnddir))
            else:
                os.mkdir(rnddir)
                print("Split {0:s}: created output/round dir: {1:s}".format(split,rnddir))

            tableFile = os.path.join(jdat['hpo_base_dir'],splitifo['tableFile'])
            tableSplitFile = os.path.join(jdat['hpo_base_dir'],splitifo['tableSplitFile'])
            trackargs = "'tableFile {0:s} tableSplitFile {1:s} paramFile {2:s}'".format(
                tableFile,tableSplitFile,prmfull)
            aptClusCmdL = [
                "~leea30/git/aptdl/APTCluster.py", 
                "-n", str(jdat['nslotsxv']),
                "--force",
                "--outdir", rnddir,
                "--bindate", jdat['bindate'],
                "--trackargs", trackargs]
            if args.action=="xvpch":
                aptClusCmdL.append("--prmpatchdir")
                aptClusCmdL.append(pchfull);
            if args.dryrun:
                aptClusCmdL.append("--dryrun");
            aptClusCmdL.extend([jdat['lblfile'],"xv"])

            aptClusCmd = " ".join(aptClusCmdL)
            print(aptClusCmd)
            resp = raw_input("Proceed? y/[n]")
            if not resp=="y":
                sys.exit("Aborted")            
            subprocess.call(aptClusCmd,shell=True)
    elif args.action=="trntrk":
        if args.roundidx:
            roundIdxs = int(args.roundidx)
            roundIdxs = range(roundIdxs,roundIdxs+1)
        else:
            lastRoundIdx,_,_,_,_ = findRoundIdx(jdat)
            print("Found latest prm/pchs => roundIdx: {0:d}".format(lastRoundIdx))
            roundIdxs = range(0,lastRoundIdx+1)

        for roundIdx in roundIdxs:
            prmfull,_,_,_ = findPrmPch(jdat,roundIdx)
            print("Found prm for roundIdx {0:d}: {1:s}".format(roundIdx,prmfull))

            for split in jdat['splits']:
                splitifo = jdat['splits'][split]
                outdir = os.path.join(jdat['hpo_base_dir'],splitifo['dir'])        
                tableFileTrn = os.path.join(jdat['hpo_base_dir'],splitifo['tableFile'])
                tableFileTrk = os.path.join(jdat['hpo_base_dir'],splitifo['testFile'])
                trackargs = "'tblFileTrn {0:s} tblFileTrk {1:s} paramFile {2:s}'".format(
                    tableFileTrn,tableFileTrk,prmfull)   
                aptClusCmdL = [
                    "~leea30/git/aptdl/APTCluster.py", 
                    "-n", str(jdat['nslotstrntrk']),
                    "--force",
                    "--outdir", outdir,
                    "--bindate", jdat['bindate'],
                    "--trackargs", trackargs]
                if args.dryrun:
                    aptClusCmdL.append("--dryrun");
                aptClusCmdL.extend([jdat['lblfile'],"trntrk"])

                aptClusCmd = " ".join(aptClusCmdL)
                print(aptClusCmd)
                resp = raw_input("Proceed? y/[n]")
                if not resp=="y":
                    sys.exit("Aborted")            
                subprocess.call(aptClusCmd,shell=True)
    else:
        sys.exit("Unrecognized action: {0:s}".format(args.action))


if __name__=="__main__":
    main()
