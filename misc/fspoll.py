#!/usr/bin/python

import sys
import os
import string

args = sys.argv
nargs = len(args)
for i in range(1,nargs,2):
    type = args[i]
    file = args[i+1]
    if type=='exists':
        if os.path.exists(file):
            val = 'y'
        else:
            val = 'n'
    elif type=='existsNE':
        if os.path.exists(file) and os.path.getsize(file)>0:
            val = 'y'
        else:
            val = 'n'
    elif type=='contents':
        if os.path.exists(file):
            with open(file,'r') as readfile:
                val = readfile.read().replace('\n','')
        else:
            val = 'DNE'

    print(val)
