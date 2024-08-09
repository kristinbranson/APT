'''
Script to track offline from python without having to load an APT project into front-end.

'''

import sys
import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
from tarfile import TarFile
import tempfile
import glob
from pathlib import Path
from datetime import datetime
import APT_interface as apt
import logging
import numpy as np

##
def parse_args(argv):

    parser = argparse.ArgumentParser(description='Track movies using the latest trained model in the APT lbl file')
    parser.add_argument("-lbl_file", help="path to APT lbl file or a directory when the lbl file has been unbundled using untar", required=True)
#    parser.add_argument('-backend',help='Backend to use for tracking. Options are docker and conda',default='docker')
    parser.add_argument('-list', help='Lists all the trained models present in the lbl file',action='store_true')
    parser.add_argument("-model_ndx", help="Use this model number (model numbers can be found using -list) instead of the latest model",type=int,default=None)
    parser.add_argument("-mov", dest="mov",help="movie(s) to track. For multi-view projects, specify movies for all the views or specify the view for the single movie using -view", nargs='+')
    parser.add_argument("-trx", dest="trx",help='trx file for movie', default=None, nargs='*')
    parser.add_argument("-log_file", dest="log_file",help='log file for output', default=None)
    parser.add_argument("-err_file", dest="err_file",help='error file for output', default=None)
    parser.add_argument('-start_frame', dest='start_frame', help='start frame for tracking', nargs='*', type=int, default=1)
    parser.add_argument('-end_frame', dest='end_frame', help='end frame for tracking', nargs='*', type=int, default=-1)
    parser.add_argument('-out', dest='out_files', help='file to save tracking results to. If track_type is "predict_track" and no predict_trk_files are specified, the pure linked tracklets will be saved to files with _pure suffix. If track_type is "predict_only" the out file will have the pure linked tracklets and predict_trk_files will be ignored.', nargs='+')
    parser.add_argument('-crop_loc', dest='crop_loc', help='crop locations given as xlo xhi ylo yhi', nargs='*', type=int, default=None)
    parser.add_argument('-view', help='track only for this view. If not specified, track for all the views', default=None, type=int)
    parser.add_argument('-stage', help='Stage for multi-stage tracking. Options are multi, first, second or None (default)', default=None)

    parser.add_argument('-track_type',choices=['predict_link','only_predict','only_link'], default='predict_link', help='for multi-animal. Whether to link the predictions or not, or only link existing tracklets. "predict_link" both predicts and links, "only_predict" only predicts but does not link, "only_link" only links existing predictions. For only_link, trk files with raw unlinked predictions must be supplied using -predict_trk_files option. Default is "predict_link"')
    parser.add_argument('-predict_trk_files', help='Intermediate trk files storing pure tracklets. Required when using link_only track_type', default=None, type=int)

    parser.add_argument('-conf_params',
                        help='conf params. These will override params from lbl file. If the model is a 2 stage tracker then this will override the params only for the first stage', default=None, nargs='*')
    parser.add_argument('-conf_params2',
                        help='conf params for 2nd stage. These will override params from lbl file for 2nd stage tracker if the model is a 2 stage tracker', default=None, nargs='*')
    
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

def get_strs(tstamp, tstamps, mtypes, tdir):
    ndx = [ix for ix, tt in enumerate(tstamps) if tt == tstamp]
    multi_stage = True if len(ndx) > 1 else False
    is_multi = [mm.startswith('multi_') for mm in mtypes]

    if multi_stage:
        first_stage = [ix for ix in ndx if is_multi[ix]][0]
        second_stage = [ix for ix in ndx if ix != first_stage][0]
        ndx = first_stage
        extra = f' -type2 {mtypes[second_stage]} -stage multi '
        if 'detect_' in mtypes[first_stage]:
            extra = ' -conf_params2 use_bbox_trx True' + extra
        else:
            extra = ' -conf_params2 use_bbox_trx False' + extra

        extra_2 = f' -trx {os.path.join(tdir, "temp.trk")} '
    else:
        ndx = ndx[0]
        extra = ''
        extra_2 = ''
        second_stage = None

    return ndx, extra, extra_2, multi_stage, second_stage


def main(argv):
    args = parse_args(argv)

    lbl_file = args.lbl_file
    del_tfile = False
    if os.path.isdir(lbl_file):
        tdir = lbl_file
    else:
        tdir = tempfile.mkdtemp()
    ## For testing
    #    lbl_file = '/groups/branson/home/kabram/APT_projects/alice_ma_20210523_tdmodel_t2.lbl'
    #    tdir = '/tmp/tmp90as'
    #    os.makedirs(tdir,exist_ok=True)

    ## Untar the label files

        print('Unbundling the label file.. ')
        tobj = TarFile.open(lbl_file)
        tobj.extractall(path=tdir)
        del_tfile = True

## Find all the stored models

    flist = glob.glob(tdir + '/*/*/view_0/*')
    mtypes = [Path(ff).parent.parent.name for ff in flist]

    tstamps_str = [Path(ff).name for ff in flist]
    tstamps = [datetime.strptime(tt,'%Y%m%dT%H%M%S') for tt in tstamps_str]
    u_tstamps = np.sort(np.unique(tstamps))
    ##
    if args.list:

        for idx, cur in enumerate(u_tstamps):
            ndx = [ix for ix, tt in enumerate(tstamps) if tt == cur]
            multi_stage = True if len(ndx) > 1 else False
            is_multi = [mm.startswith('multi_') or mm.startswith('detect_') for mm in mtypes]

            if multi_stage:
                first_stage = [ix for ix in ndx if is_multi[ix]][0]
                second_stage = [ix for ix in ndx if ix != first_stage][0]
                print(f'* Model {idx+1} -- multi-animal multi-stage with {get_pretty_name(mtypes[first_stage])} and {get_pretty_name(mtypes[second_stage])} networks, trained at {cur.strftime("%Y %b %m %H:%M")}')
            else:
                mstr = 'multi-animal' if is_multi[ndx[0]] else 'single-animal'
                print(f'* Model {idx+1} -- {mstr} {get_pretty_name(mtypes[ndx[0]])} network, trained at {cur.strftime("%Y %b %m %H:%M")}')
        if del_tfile:
            shutil.rmtree(tdir)
        return

    if args.model_ndx is None:
        m_stamp = max(tstamps)
    else:
        m_stamp = u_tstamps[args.model_ndx-1]

    ndx, extra, extra_2, multi_stage, second_stage = get_strs(m_stamp, tstamps,mtypes,tdir)

    stripped_lbls = glob.glob(tdir + f'/*/{tstamps_str[ndx]}_*.lbl')
    usestrippedlbl = len(stripped_lbls) > 0
    if usestrippedlbl:
        config_file = stripped_lbls[0]
    else:
        json_configs = glob.glob(tdir + f'/*/{tstamps_str[ndx]}_*.json')
        assert len(json_configs) > 0, f'Could not find either json config or stripped lbl file'
        config_file = json_configs[0]
        #model_files = glob.glob()

    pre_str = ''
    if args.view is not None:
        pre_str += f' -view {args.view}'
        tndx = argv.index('-view')
        argv.pop(tndx); argv.pop(tndx)

    if args.conf_params is not None:
        c_str = ' '.join(args.conf_params)
        pre_str += f' -conf_params {c_str}'
        tndx = argv.index('-conf_params')
        argv.pop(tndx)
        while (len(argv)>tndx) and (not argv[tndx].startswith('-')):
            argv.pop(tndx)

    if args.conf_params2 is not None:
        c_str = ' '.join(args.conf_params2)
        pre_str += f' -conf_params2 {c_str}'
        tndx = argv.index('-conf_params2')
        argv.pop(tndx)
        while (len(argv)>tndx) and (not argv[tndx].startswith('-')):
            argv.pop(tndx)

    if args.log_file is not None:
        pre_str += f' -log_file {args.log_file}'
        tndx = argv.index('-log_file')
        argv.pop(tndx)
        argv.pop(tndx)

    if args.err_file is not None:
        pre_str += f' -err_file {args.err_file}'
        tndx = argv.index('-err_file')
        argv.pop(tndx)
        argv.pop(tndx)

    # if usestrippedlbl:
    #     cmd = f'{config_file} '
    # else:
    #     cmd = ''
    cmd = f'{config_file} -type {mtypes[ndx]} -cache {tdir} {pre_str} {extra} -name {tstamps_str[ndx]} track {extra_2}'


##
    a_argv = cmd.split() + argv
    if '-model_ndx' in a_argv:
        tndx = a_argv.index('-model_ndx')
        a_argv.pop(tndx); a_argv.pop(tndx)
    if '-lbl_file' in a_argv:
        tndx = a_argv.index('-lbl_file')
        a_argv.pop(tndx); a_argv.pop(tndx)

    pstr = get_pretty_name(mtypes[ndx])
    dstr = f'{tstamps[ndx]}'
    mstr = 'multi-stage ' if multi_stage else ''
    str1 = f'Tracking using {mstr}{pstr}'
    if multi_stage:
        str1 = f' {str1} (first-stage) and {get_pretty_name(mtypes[second_stage])} models'
    else:
        str1 = f'{str1} model'
    str1 = f'{str1} trained on {dstr}'
    print('------------------------------------')
    print('------------------------------------')
    print('')
    print(str1)
    print('')
    print('------------------------------------')
    print('------------------------------------')
    apt.main(a_argv)

    if del_tfile:
        shutil.rmtree(tdir)

##
if __name__ == "__main__":
    main(sys.argv[1:])



if False:
    ##
    in_cmd = '/groups/branson/home/kabram/APT_projects/alice_ma_20210523_tdmodel_t2.lbl -mov /groups/branson/home/kabram//flyMovies/apt/grooming_GMR_30B01_AE_01_CsChr_RigB_20150903T161828/movie.ufmf -start_frame 300 -end_frame 350 -out /groups/branson/home/kabram/temp/kk.trk'
    argv = in_cmd.split()
