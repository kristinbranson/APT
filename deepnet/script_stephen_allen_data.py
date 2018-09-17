##
import APT_interface
import h5py
import  tensorflow as tf
from multiResData import *
import localSetup
import os
import  PoseUNet
import  subprocess
import copy
import json
import copy
from matplotlib import cm
import matplotlib.pyplot as plt
import APT_interface
import datetime
import ast
import argparse
import pickle
import hdf5storage

# split_type = 'easy'
extra_str = '_normal_bnorm'
# extra_str = '_my_bnorm'
# dropout = True
imdim = 1

def get_name(args):
    return 'unet_' + args['name']
    # cur_str = 'unet' + extra_str
    # if bool(args['dropout']):
    #     cur_str += '_do'  # dropout during training
    # return cur_str


def get_cache_dir(conf, split, split_type):
    outdir = os.path.join(conf.cachedir, 'cv_split_{}'.format(split))
    if split_type == 'easy':
        outdir += '_easy'
    else:
        outdir += '_hard'
    return outdir


def get_conf(pdict):
    # Fixed
    view = int(pdict['view'])
    if view == 0:
        from stephenHeadConfig import sideconf as conf
        curconf = copy.deepcopy(conf)
        curconf.imsz = (350, 230)
    else:
        from stephenHeadConfig import conf as conf
        curconf = copy.deepcopy(conf)
        curconf.imsz = (350, 350)
    curconf.batch_size = 4
    curconf.unet_rescale = 1
    curconf.imgDim = 1
    split_num = int(pdict['split_num'])
    split_type = pdict['split_type']
    curconf.cachedir = get_cache_dir(conf, split_num, split_type)

    for k in pdict.keys():
        if k in ['view','split_num','split_type','restore']:
            continue
        try:
            curval = ast.literal_eval(pdict[k])
        except ValueError:
            curval = pdict[k]
        setattr(curconf,k,curval)

    return  curconf


def create_tfrecords(args):
    data_file = '/groups/branson/bransonlab/mayank/PoseTF/headTracking/trnDataSH_20180503_notable.mat'
    split_file = '/groups/branson/bransonlab/apt/experiments/data/trnSplits_20180509.mat'

    D = h5py.File(data_file,'r')
    S = APT_interface.loadmat(split_file)

    nims = D['IMain_crop2'].shape[1]
    movid = np.array(D['mov_id']).T - 1
    frame_num = np.array(D['frm']).T - 1
    if args['split_type'] == 'easy':
        split_arr = S['xvMain3Easy']
    else:
        split_arr = S['xvMain3Hard']


    for view in range(2):
        for split in range(3):
            args['view'] = view
            args['split_num'] = split
            conf = get_conf(args)
            if not os.path.exists(conf.cachedir):
                os.mkdir(conf.cachedir)

            outdir = conf.cachedir
            train_filename = os.path.join(outdir,conf.trainfilename)
            val_filename = os.path.join(outdir,conf.valfilename)
            env = tf.python_io.TFRecordWriter(train_filename + '.tfrecords')
            val_env = tf.python_io.TFRecordWriter(val_filename + '.tfrecords')
            splits = [[], []]
            all_locs = np.array(D['xyLblMain_crop2'])[view,:,:,:]
            for indx in range(nims):
                cur_im = np.array(D[D['IMain_crop2'][view,indx]]).T
                cur_locs = all_locs[...,indx].T
                mov_num = movid[indx]
                cur_frame_num = frame_num[indx]

                if split_arr[indx,split] == 1:
                    cur_env = val_env
                    splits[1].append([indx,cur_frame_num[0],0])
                else:
                    cur_env = env
                    splits[0].append([indx,cur_frame_num[0],0])


                im_raw = cur_im.tostring()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': int64_feature(cur_im.shape[0]),
                    'width': int64_feature(cur_im.shape[1]),
                    'depth': int64_feature(1),
                    'trx_ndx': int64_feature(0),
                    'locs': float_feature(cur_locs.flatten()),
                    'expndx':float_feature(mov_num),
                    'ts':float_feature(cur_frame_num[0]),
                    'image_raw':bytes_feature(im_raw)
                }))


                cur_env.write(example.SerializeToString())
            env.close()
            val_env.close()
            with open(os.path.join(outdir, 'splitdata.json'), 'w') as f:
                json.dump(splits, f)

    D.close()

def train(args):
    split_num = int(args['split_num'])
    view = int(args['view'])
    split_type = args['split_type']
    if 'restore' in args.keys():
        restart = ast.literal_eval(args['restore'])
    else:
        restart = False
    conf = get_conf(args)
    print('split:{}, view:{} split_type:{}'.format(split_num,view,split_type))
    self = PoseUNet.PoseUNet(conf,name=get_name(args))
    self.train_unet(restart,0)


def submit_jobs(args,views=range(2)):

    for view in views:
        for split_num in range(3):
            args['view'] = view
            args['split_num'] = split_num
            curconf = get_conf(args)
            name = get_name(args)
            sing_script = os.path.join(curconf.cachedir,'singularity_script_{}.sh'.format(name))
            sing_log = os.path.join(curconf.cachedir,'singularity_script_{}.log'.format(name))
            sing_err = os.path.join(curconf.cachedir,'singularity_script_{}.err'.format(name))
            arg_str = ''
            for k in args.keys():
                arg_str = '{} -{} {}'.format(arg_str,k,args[k])
            with open(sing_script, 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('. /opt/venv/bin/activate\n')
                f.write('cd /groups/branson/home/kabram/PycharmProjects/poseTF\n')
                f.write('python script_stephen_allen_data.py {}'.format(arg_str))
                f.write('\n')

            os.chmod(sing_script, 0755)

            cmd = '''ssh 10.36.11.34 '. /misc/lsf/conf/profile.lsf; bsub -oo {} -eo {} -n2 -gpu "num=1" -q gpu_any "singularity exec --nv /misc/local/singularity/branson_v2.simg {}"' '''.format(
                sing_log, sing_err, sing_script)  # -n2 because SciComp says we need 2 slots for the RAM
            subprocess.call(cmd, shell=True)
            print('Submitted jobs for batch split:{} view:{}'.format(split_num,view))


def compile_results(args,views=range(2)):
    all_dist = [[],[]]
    rims = [[],[]]
    rlocs = [[],[]]
    rinfo = [[],[]]
    rpredlocs = [[],[]]
    for view in views:
        for split_num in range(3):
            args['view'] = view
            args['split_num'] = split_num
            curconf = get_conf(args)
            tf.reset_default_graph()
            self = PoseUNet.PoseUNet(curconf,name=get_name(args))
            dist, ims, preds, predlocs, locs, info = self.classify_val(0)
            all_dist[view].append(dist)
            rims[view].append(ims)
            rlocs[view].append(locs)
            rinfo[view].append(info)
            rpredlocs[view].append(predlocs)
        all_dist[view] = np.concatenate(all_dist[view],0)
        rims[view] = np.concatenate(rims[view],0)
        rlocs[view] = np.concatenate(rlocs[view],0)
        rinfo[view] = np.concatenate(rinfo[view], 0)
        rpredlocs[view] = np.concatenate(rpredlocs[view], 0)
    return all_dist, rims, rlocs, rinfo, rpredlocs


def save_data(name='normal'):
    for split_type in ['easy','hard']:
        args = {'name': name,
                'split_num': 2,
                'split_type': split_type,
                'view': 1}
        all_dist, _, locs, info, predlocs = compile_results(args)
        fname = '/groups/branson/home/kabram/bransonlab/PoseTF/headTracking/{}_{}_cv_data.p'.format(name,args['split_type'])
        with open(fname,'w') as f:
            pickle.dump([all_dist,locs, info,predlocs],f)

    convert_saved_to_matlab(name)

def convert_saved_to_matlab(name='normal'):
    data = {}
    for split_type in ['easy','hard']:
        fname = '/groups/branson/home/kabram/bransonlab/PoseTF/headTracking/{}_{}_cv_data.p'.format(name,split_type)
        with open(fname,'r') as f:
            all_dist, locs, info, predlocs = pickle.load(f)
            data['dist_{}_side'.format(split_type)] = all_dist[0]
            data['dist_{}_front'.format(split_type)] = all_dist[1]
            data['locs_{}_side'.format(split_type)] = locs[0]
            data['locs_{}_front'.format(split_type)] = locs[1]
            data['info_{}_side'.format(split_type)] = info[0]
            data['info_{}_front'.format(split_type)] = info[1]
            data['pred_{}_side'.format(split_type)] = predlocs[0]
            data['pred_{}_front'.format(split_type)] = predlocs[1]
    fname = '/groups/branson/home/kabram/bransonlab/PoseTF/headTracking/{}_cv_data.mat'.format(name)
    hdf5storage.savemat(fname,data)

def create_result_images(split_type):
    dist, ims, locs = compile_results(split_type)
    in_locs = copy.deepcopy(locs)

    dstr = datetime.datetime.now().strftime('%Y%m%d')

    for view in range(2):
        perc = np.percentile(dist[view], [75, 90, 95, 98, 99, 99.5], axis=0)

        f, ax = plt.subplots()
        im = ims[view][0, :, :, 0]
        locs = in_locs[view][0, ...]

        ax.imshow(im) if im.ndim == 3 else ax.imshow(im, cmap='gray')
        # ax.scatter(locs[:,0],locs[:,1],s=20)
        cmap = cm.get_cmap('jet')
        rgba = cmap(np.linspace(0, 1, perc.shape[0]))
        for pndx in range(perc.shape[0]):
            for ndx in range(locs.shape[0]):
                ci = plt.Circle(locs[ndx, :], fill=False,
                                radius=perc[pndx, ndx], color=rgba[pndx, :])
                ax.add_artist(ci)
            plt.axis('off')
            f.savefig('/groups/branson/home/kabram/temp/stephen_unet_{}_view{}_results_{}.png'.format(dstr,view, pndx),
                      bbox_inches='tight', dpi=400)


def main(argv):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-split_num", dest='split_num',required=True,type=int)
    # parser.add_argument('-dropout',dest='dropout',action="store_true")
    # parser.add_argument('-view',dest='view',required=True,type=int)
    # parser.add_argument('-split_type',dest='split_type',required=True)
    # args = parser.parse_args(argv)
    print argv
    pdict = {}
    assert len(argv)%2 == 0, 'Number of params should be even'
    for ndx in range(0,len(argv),2):
        assert argv[ndx][0] == '-', 'Odd Arguments should start with -'
        pdict[argv[ndx][1:]] = argv[ndx+1]
    train(pdict)

if __name__ == "__main__":
    main(sys.argv[1:])
