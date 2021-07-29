from __future__ import division
from __future__ import print_function

import sys
import os
import platform
import datetime
import json
import argparse
import cv2

import movies

def parse_args(argv):
    '''
    videotest -gencompare /path/to/existing/vidtestdir
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-gencompare',
                        dest='gencompare',
                        required=True,
                        help='Full path to existing videotest dir')
    args = parser.parse_args(argv)
    
    return args


def get_jsondata(mov, nmax, frms):
    pyver = platform.python_version()
    uname = platform.uname()
    osname = uname.system
    hostname = uname.node
    now_str = datetime.datetime.today().strftime('%Y%m%dT%H%M%S')
    frms1b = [x+1 for x in frms]
    jsd = {
        'nmaxsupplied': True,
        'nmaxused': nmax,
        'host': hostname,
        'os': osname,
        'pyver': pyver,
        'nowstr': now_str,
        'mov': mov,
        'frms': frms1b,
        }
    return jsd
    
        
def get_testdir(mov,testdir0,jsdata):
    movP, movF = os.path.split(mov)
    movF, movE = os.path.splitext(movF)
    outdir = 'VideoTest_{}_{}_{}_py{}_{}'.format(
        movF,jsdata['os'],jsdata['host'],jsdata['pyver'],jsdata['nowstr'])
    outdir = os.path.join(movP,outdir);
    return outdir
    
def read_ims(mov, frms):
    '''
    Read the frames in frms (0-based) and return in a list
    '''
    cap = movies.Movie(mov)
    ims = [cap.get_frame_unbuffered(i)[0] for i in frms]
    cap.close()
    return ims
    
def read_seq(mov, nmax):
    '''
    Read the frames 0..nmax-1 and sequentially and return a list
    '''
    return read_ims(mov, range(nmax))
    

def main(argv):
    args = parse_args(argv)

    vtdir = args.gencompare
    jsf = os.path.join(vtdir,'info.json')
    with open(jsf,'r') as f:
        jsdata = json.load(f)

    mov = jsdata['mov']
    if not os.path.isabs(mov):
        vtparentdir = os.path.dirname(vtdir)
        mov = os.path.join(vtparentdir,mov)
        print("Relative moviename specified in info.json. Expanding to {}".format(mov))

    frms1b = jsdata['frms']  # json is written as 1-based from matlab
    frms0b = [x-1 for x in frms1b] 

    cap = movies.Movie(mov)
    if jsdata['nmaxsupplied']:
        nmax = jsdata['nmaxused']
        print("nmax={}")
    else:
        nmax = cap.get_n_frames()
        print("nmax not supplied, using nmax={}".format(nmax))
    cap.close()    
    # nmax is 1-past-end of 0-based frames

    assert all([x < nmax for x in frms0b]),"One or more frames out of range given nmax."

    ims_sq = read_seq(mov, nmax)
    ims_ra = read_ims(mov, frms0b)

    jsdataout = get_jsondata(mov,nmax,frms0b)
    vtdirout = get_testdir(mov,vtdir,jsdataout)
    assert not os.path.exists(vtdirout)
    os.mkdir(vtdirout)
    print("Made output dir {}".format(vtdirout))

    for ndx, im in enumerate(ims_sq):
        ndx1b = ndx+1
        fname = 'sr_{:06d}.png'.format(ndx1b)
        fname = os.path.join(vtdirout,fname)
        cv2.imwrite(fname,im)
    print("Wrote SRs")

    for ndx, im in enumerate(ims_ra):
        ndx1b = ndx+1
        fname = 'rar_{:06d}_{:06d}.png'.format(ndx1b,frms1b[ndx])
        fname = os.path.join(vtdirout,fname)
        cv2.imwrite(fname,im)
    print("Wrote RARs")

    fname = os.path.join(vtdirout,'info.json');
    with open(fname,'w') as f:
        json.dump(jsdataout,f)
    print("Wrote {}".format(fname))
    
        
    
if __name__ == "__main__":
    main(sys.argv[1:])


