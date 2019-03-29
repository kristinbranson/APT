data_type = 'alice'


##
import APT_interface as apt
import h5py
import PoseTools

if data_type == 'alice':
    lbl_file = '/groups/branson/bransonlab/apt/experiments/data/multitarget_bubble_expandedbehavior_20180425_FxdErrs_OptoParams20181126_dlstripped.lbl'
elif data_type == 'stephen':
    lbl_file = '/groups/branson/bransonlab/apt/experiments/data/sh_trn4992_gtcomplete_cacheddata_dlstripped.lbl'
else:
    lbl_file = ''


lbl = h5py.File(lbl_file,'r')
nviews = int(apt.read_entry(lbl['cfg']['NumViews']))
lbl.close()
cache_dir = '/nrs/branson/mayank/apt_cache'
all_models = ['deeplabcut','mdn','unet','openpose']

## create dbs

for view in range(nviews):
    for tndx in range(len(all_models)):
        train_type = all_models[tndx]
        conf = apt.create_conf(lbl_file,view,'apt_expt',cache_dir,train_type)
        if train_type == 'deeplabcut':
            apt.create_deepcut_db(conf,split=False,use_cache=True)
        elif train_type == 'leap':
            apt.create_leap_db(conf,split=False,use_cache=True)
        else:
            apt.create_tfrecord(conf,split=False,use_cache=True)


##
# gpu_model = 'TeslaV100_SXM2_32GB'
gpu_model = 'GeForceRTX2080Ti'
train_type = 'deeplabcut'
sdir = '/groups/branson/home/kabram/bransonlab/APT/deepnet/singularity_stuff'
common_conf = {}
common_conf['rrange'] = 10
common_conf['trange'] = 5
common_conf['mdn_use_unet_loss'] = True
common_conf['dl_steps'] = 60000
common_conf['decay_steps'] = 20000
common_conf['save_step'] = 5000
common_conf['batch_size'] = 8

other_conf = [{'dlc_augment':True},{'dlc_augment':False,'dl_steps':300000}]
cmd_str = ['dlc_aug','dlc_noaug']


for view in range(nviews):

    for conf_id in range(len(other_conf)):

        common_cmd = 'APT_interface.py {} -name apt_expt -cache {}'.format(lbl_file,cache_dir)
        end_cmd = 'train -skip_db -use_cache'
        cmd_opts = {}
        cmd_opts['type'] = train_type
        cmd_opts['train_name'] = cmd_str[conf_id]
        conf_opts = common_conf.copy()
        conf_opts.update(other_conf[conf_id])

        if len(conf_opts) > 0:
            conf_str = ' -conf_params'
            for k in conf_opts.keys():
                conf_str = '{} {} {} '.format(conf_str,k,conf_opts[k])
        else:
            conf_str = ''

        opt_str = ''
        for k in cmd_opts.keys():
            opt_str = '{} -{} {} '.format(opt_str,k,cmd_opts[k])

        cur_cmd = common_cmd + conf_str + opt_str + end_cmd
        print cur_cmd
        print
        cmd_name = 'alice_view{}_{}'.format(view,cmd_str[conf_id])
        PoseTools.submit_job(cmd_name, cur_cmd, sdir, gpu_model=gpu_model)

