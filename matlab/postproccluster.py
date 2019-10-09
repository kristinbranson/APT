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
import time
import numpy as np
from threading import Timer

#from APTCluster import makecmd

DEFAULTAPTDIR = os.path.dirname(os.path.realpath(__file__))

MATLABCMDPAT = "{mlbin} -nodisplay -r \"try, cd('{aptroot}'); APT.setpath; run3dtriangulate('{trkfile1}','{trkfile2}','{pptype}','crigmat','{crigmat}'); exit; catch ME, disp('Error caught:'); disp(getReport(ME)); disp('Exiting.'); exit; end;\""

class ArgDefaultsAndRawDescHelpFormatter(argparse.HelpFormatter):
    """
    Combination of RawDescriptionHelpFormatter and ArgumentDefaultsHelpFormatter.

    It appears HelpFormatter is not explicitly designed to be publicly subclassed
    so this may be somewhat fragile
    """

    def _fill_text(self, text, width, indent):
        return ''.join([indent + line for line in text.splitlines(True)])

    def _get_help_string(self, action):
        help = action.help
        if '%(default)' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += ' (default: %(default)s)'
        return help


def main():
    # https://github.com/carlobaldassi/ArgParse.jl/issues/12
    epilogstr = """
    Examples:
        .../postproccluster.py -h
        .../postproccluster.py --trk1 view1.trk --trk2 view2.trk calrig.mat (run in dir containing view1.trk, view2.trk, and calrig.mat)
        .../postproccluster.py --trklistfile trkpairs.txt calrig.mat (run in dir containing trkpairs.txt and calrig.mat)
    """
    # epilogstr = 'Examples:\n.../postproccluster.py -h\n .../postproccluster.py --trk1 view1.trk --trk2 view2.trk calrig.mat (run in dir containing view1.trk, view2.trk, and calrig.mat)\n.../postproccluster.py --trklistfile trkpairs.txt calrig.mat (run in dir containing trkpairs.txt and calrig.mat)\n'

# ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=ArgDefaultsAndRawDescHelpFormatter,
                                     epilog=epilogstr)

    parser.add_argument("crig",
                        help="Matfile (short) name containing calibration (CalRig) object")
    parser.add_argument("--dataroot",
                        help="Data folder containing trkfiles (or trklistfile) and crig",
                        default=os.getcwd())
    parser.add_argument("--trk1",
                        help="Trkfile (short) name for view 1")
    parser.add_argument("--trk2",
                        help="Trkfile (short) name for view 2")
    parser.add_argument("--trklistfile",
                        help="CSV with trkfile pairs")
    parser.add_argument("--ppalg",
                        help="3D Postprocessing algorithm",
                        choices=["experimental","triangulate"],
                        default="experimental")
    parser.add_argument("--mlbin",
                        help="Full path to matlab binary",
                        default='/misc/local/matlab-2019a/bin/matlab')
    parser.add_argument("--aptroot",
                        help="Full path to APT root",
                        default=DEFAULTAPTDIR)
    parser.add_argument("-n", "--nslots",
                        help="number of cluster slots",
                        default="1") #, metavar="NSLOTS")
    parser.add_argument("-l", "--logdir",
                        help="location to output bsub script and logs. Defaults to dataroot",
                        default=os.getcwd())
    parser.add_argument("--account",
                        help="account to bill for cluster time",
                        default="") #, metavar="ACCOUNT")
    parser.add_argument("--dryrun",
                        help="Show but do not execute cluster (bsub) commands. Code generation still occurs.",
                        action="store_true", default=False)


    args = parser.parse_args()

    if args.trk1 and args.trk2:
        tfilesfull = [(os.path.join(args.dataroot, args.trk1), os.path.join(args.dataroot, args.trk2))]
    elif args.trklistfile:
        trklistfile = os.path.join(args.dataroot, args.trklistfile)
        with open(trklistfile) as tlf:
            rdr = csv.reader(tlf, delimiter=',')
            tfilesfull = map(tuple, rdr)
    else:
        estr = 'Either --trk1 and --trk2 must be specified or --trklistfile must be specified.'
        sys.exit(estr)

    args.KEYWORD = "apt"  # used for log/sh filenames, sge job name
    args.TMP_ROOT_DIR = "/scratch/`whoami`"
    bsubargs = "-n " + args.nslots
    if args.account:
        bsubargs = "-P {0:s} ".format(args.account) + bsubargs

    crigfull = os.path.join(args.dataroot, args.crig)

    for tf1, tf2 in tfilesfull:
        if not os.path.exists(tf1):
            sys.exit('Cannot find trkfile {}.'.format(tf1))
        if not os.path.exists(tf2):
            sys.exit('Cannot find trkfile {}.'.format(tf2))

    files = [crigfull, args.logdir]
    filetys = ['calibration object matfile', 'logdir']
    for f, ty in zip(files, filetys):
        if not os.path.exists(f):
            estr = 'Cannot find {} {}.'.format(ty, f)
            sys.exit(estr)

    singlepair = len(tfilesfull) == 1
    nowstr = datetime.datetime.now().strftime("%Y%m%dT%H%M%S%f")
    nowstr = nowstr[:-3]  # keep only milliseconds

    for idx, tfs in enumerate(tfilesfull):
        # jobid
        if singlepair:
            jobid = args.KEYWORD + "-" + nowstr
        else:
            jobid = args.KEYWORD + "-{0:04}-".format(idx) + nowstr
        print(jobid)
        shfile = os.path.join(args.logdir, "{0:s}.sh".format(jobid))
        logfile = os.path.join(args.logdir, "{0:s}.log".format(jobid))

        #cmd = makecmd([args.projfile, args.action, args.mov, ''], usecompiled=False)
        cmd = MATLABCMDPAT.format(mlbin=args.mlbin, aptroot=args.aptroot,
                                  trkfile1=tfs[0], trkfile2=tfs[1], pptype=args.ppalg, crigmat=crigfull)
        #print(cmd)
        gencode(shfile, cmd)

        qargs = '{0:s}  -o {1:s} -J {2:s} {3:s}'.format(bsubargs, logfile, jobid, shfile)
        qsubcmd = "bsub " + qargs

        print(qsubcmd)
        if not args.dryrun:
            subprocess.call(qsubcmd, shell=True)

    sys.exit()


def gencode(fname, cmd):
    f = open(fname, 'w')
    print("#!/bin/bash", file=f)
    print("", file=f)
    print("source ~/.bashrc", file=f)
    print("umask 002", file=f)
    print("unset DISPLAY", file=f)
    print("", file=f)
    print(cmd, file=f)
    print("", file=f)
    f.close()
    os.chmod(fname, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH);


# def makecmd(s, cmd='', usecompiled=False):
#     if isinstance(s, list):
#         for ss in s:
#             cmd = makecmd(ss, cmd, usecompiled)
#         return cmd
#
#     if usecompiled:
#         if not isinstance(s, str):
#             s = str(s)
#         if len(s) == 0:
#             s = "''"
#         if len(cmd) > 0:
#             cmd = cmd + ' '
#         cmd = cmd + s
#     else:
#         if len(cmd) > 0:
#             cmd = cmd + ','
#         if isinstance(s, str) and ((len(s) < 1) or s[0] != "'"):
#             cmd = cmd + "'" + s + "'"
#         else:
#             cmd = cmd + str(s)
#     return cmd


if __name__ == "__main__":
    main()

