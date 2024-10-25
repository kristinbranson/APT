#! /usr/bin/env python3

# FutureWarnings from import corrupting fspoll output/parse
# alternatively, could eg prefix each 'real' output line with
# a sentinel or keyword
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import os
import string
import re
import glob
#import numpy as np
#import h5py


args = sys.argv
nargs = len(args)
for i in range(1,nargs,2):
    ty = args[i]
    file = args[i+1]
    if ty=='exists':
        if os.path.exists(file):
            val = 'y'
        else:
            val = 'n'
    elif ty=='existsNE':
        if os.path.exists(file) and os.path.getsize(file)>0:
            val = 'y'
        else:
            val = 'n'
    elif ty=='existsNEerr':
        val = 'n'
        if os.path.exists(file) and os.path.getsize(file)>0:
            with open(file,'r') as readfile:
                contents = readfile.read()
                if re.search('exception',contents,flags=re.IGNORECASE):
                    val = 'y'
    elif ty=='contents':
        if os.path.exists(file):
            with open(file,'r') as readfile:
                val = readfile.read().replace('\n','')
        else:
            val = 'DNE'
    elif ty=='lastmodified':
        if os.path.exists(file):
            statbuf = os.stat(file)
            val = statbuf.st_mtime
            val = str(val)
        else:
            val = 'DNE'
#     elif ty=='nfrmtracked':
#         if os.path.exists(file):
#             try:
#                 mat = h5py.File(file,'r')
#                 canCount = True
#                 if 'pTrk' in mat.keys():
#                     f = 'pTrk'
#                 elif 'pred_locs' in mat.keys():
#                     f = 'pred_locs'
#                 else:
#                     val = 'DNE'
#                     canCount = False
#                 if canCount:
#                     pTrk = mat[f].value
#                     if pTrk.ndim==4:
#                         val = str(np.count_nonzero(~np.isnan(pTrk[:,:,0,0])))
#                     else:
#                         val = str(np.count_nonzero(~np.isnan(pTrk[:,0,0])))
# 
#             except IOError:               
#                 # txt file with number of frames
#                 with open(file,'r') as fid:
#                     l = fid.read()
#                     val = int(l)
                
                
    elif ty=='mostrecentmodel':
        # file is dir containing models

        REPAT = 'deepnet-(?P<iter>\d+)'
        reobj = re.compile(REPAT)        
        globpat = os.path.join(file,'deepnet-*')        
        datafiles = glob.glob(globpat)

        maxiter = -1
        for d in datafiles:
            dbase = os.path.basename(d)
            m = reobj.match(dbase)
            if m:
                iterf = m.group('iter')
                iterf = int(iterf) 
                if iterf>maxiter:
                    maxiter = iterf
                    
        if maxiter==-1:
            val = 'DNE'
        else:
            val = maxiter
            val = str(val)
            
        
    print(val)
