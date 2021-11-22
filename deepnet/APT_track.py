'''
Script to track offline from python without having to load an APT project into front-end.

'''

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
from tarfile import TarFile
import tempfile
import glob
from pathlib import Path
from datetime import datetime
import APT_interface as apt

##
def parse_args(argv):

    parser = argparse.ArgumentParser(description='Track movies using the latest trained model in the APT lbl file')
    parser.add_argument("lbl_file", help="path to APT lbl file")
#    parser.add_argument('-backend',help='Backend to use for tracking. Options are docker and conda',default='docker')
    parser.add_argument("-mov", dest="mov",help="movie(s) to track. For multi-view projects, specify movies for all the views or specify the view for the single movie using -view", nargs='+')  # KB 20190123 removed required because list_file does not require mov
    parser.add_argument("-trx", dest="trx",help='trx file for movie', default=None, nargs='*')
    parser.add_argument('-start_frame', dest='start_frame', help='start frame for tracking', nargs='*', type=int, default=1)
    parser.add_argument('-end_frame', dest='end_frame', help='end frame for tracking', nargs='*', type=int, default=-1)
    parser.add_argument('-out', dest='out_files', help='file to save tracking results to', required=True, nargs='+')
    parser.add_argument('-crop_loc', dest='crop_loc', help='crop locations given as xlo xhi ylo yhi', nargs='*', type=int, default=None)
    parser.add_argument('-view', dest='view', help='track only for this view. If not specified, track for all the views', default=None, type=int)

    args = parser.parse_args(argv)
    return args

def get_latest(flist):
    return None

def get_pretty_name(model_type):
    if model_type == 'mdn_joint_fpn':
        pstr = 'Single Animal GRONe'
    elif model_type == 'multi_mdn_joint_torch':
        pstr = 'Multi Animal GRONe'
    elif model_type == 'detect_mmdetect':
        pstr = 'DETR object detection'
    elif model_type == 'mmpose':
        pstr = 'Single Animal MSPN'
    elif model_type == 'openpose':
        pstr = 'Open Pose'
    elif model_type == 'multi_openpose':
        pstr = 'Open Pose'
    else:
        pstr = model_type

    return pstr

def main(argv):
    args = parse_args(argv)
    lbl_file = args.lbl_file
    tdir = tempfile.mkdtemp()
## For testing
#    lbl_file = '/groups/branson/home/kabram/APT_projects/alice_ma_20210523_tdmodel_t2.lbl'
#    tdir = '/tmp/tmp90as'
#    os.makedirs(tdir,exist_ok=True)

## Untar the label files

    tobj = TarFile.open(lbl_file)
    tobj.extractall(path=tdir)

## Find all the stored models

    flist = glob.glob(tdir + '/*/*/*/*')
    mtypes = [Path(ff).parent.parent.name for ff in flist]

    tstamps_str = [Path(ff).name for ff in flist]
    tstamps = [datetime.strptime(tt,'%Y%m%dT%H%M%S') for tt in tstamps_str]
    latest = max(tstamps)
##
    ndx = [ix for ix, tt in enumerate(tstamps) if tt==latest]
    multi_stage = True if len(ndx)>1 else False
    is_multi = [mm.startswith('multi_') for mm in mtypes]

    if multi_stage:
        first_stage = [ix for ix in ndx if is_multi[ix]][0]
        second_stage = [ix for ix in ndx if ix!=first_stage][0]
        ndx = first_stage
        extra = f' -type2 {mtypes[second_stage]} -stage multi '
        if 'detect_' in mtypes[first_stage]:
            extra = ' -conf_params2 use_bbox_trx True' + extra
        else:
            extra = ' -conf_params2 use_bbox_trx False' + extra

        extra_2 = f' -trx {os.path.join(tdir,"temp.trk")} '
    else:
        ndx = ndx[0]
        extra = ''
        extra_2 = ''

    stripped_lbl = glob.glob(tdir + f'/*/{tstamps_str[ndx]}_*.lbl')[0]
    cmd = f'{stripped_lbl} -type {mtypes[ndx]} -cache {tdir} -name {tstamps_str[ndx]} {extra} track {extra_2}'

##
    a_argv = cmd.split() + argv[1:]
    pstr = get_pretty_name(mtypes[ndx])
    dstr = f'{tstamps[ndx]}'
    str = f'Tracking using {pstr}'
    if multi_stage:
        str = f'{str} and {get_pretty_name(mtypes[second_stage])} models'
    else:
        str = f'{str} model'
    str = f'{str} trained on {dstr}'
    print('------------------------------------')
    print('------------------------------------')
    print('')
    print(str)
    print('')
    print('------------------------------------')
    print('------------------------------------')
    apt.main(a_argv)


##
if __name__ == "__main__":
    main(sys.argv[1:])



if False:
    ##
    in_cmd = '/groups/branson/home/kabram/APT_projects/alice_ma_20210523_tdmodel_t2.lbl -mov /groups/branson/home/kabram//flyMovies/apt/grooming_GMR_30B01_AE_01_CsChr_RigB_20150903T161828/movie.ufmf -start_frame 300 -end_frame 350 -out /groups/branson/home/kabram/temp/kk.trk'
    argv = in_cmd.split()