import PoseTools as pt
import sys
import os
import deeplabcut
import APT_interface as apt
import tempfile
import h5py
from leap.training import train as leap_train
import leap.training


leap_orig_dir  = '/groups/branson/bransonlab/mayank/apt_expts/leap_orig2'
main_script = 'leap/training.py'

class Pose_leap_orig(object):

    def __init__(self,conf,name='deepnet'):
        self.name = name
        self.conf = conf

    def train_wrapper(self,restore=False):
        conf = self.conf
        fname = os.path.join(conf.cachedir,'run_leap_orig_{}.sh'.format(pt.datestr()))
        in_file = os.path.join(conf.cachedir,'leap_train.h5')

        conf.use_leap_preprocessing = True
        leap_train(data_path=os.path.join(conf.cachedir, 'leap_train.h5'),
                   base_output_path=conf.cachedir,
                   run_name=self.name,
                   # net_name=conf.leap_net_name,
                   box_dset="box",
                   confmap_dset="joints",
                   rotate_angle=conf.rrange,
                   epochs=conf.dl_steps // conf.display_step,
                   batch_size=conf.batch_size,
                   batches_per_epoch=conf.display_step,
                   conf=conf)

        # out_file = os.path.join(conf.cachedir,'leap_train_orig.h5')
        # ii = h5py.File(in_file, 'r')
        # ims = ii['box'][:]
        # locs = ii['joints'][:]
        # eid = ii['exptID'][:]
        # frid = ii['framesIdx'][:]
        # tid = ii['trxID'][:]
        # ii.close()
        #
        # hmaps = pt.create_label_images(locs, conf.imsz[:2], 1, 5)
        # hmaps = (hmaps + 1) / 2  # brings it back to [0,1]
        #
        # hf = h5py.File(out_file, 'w')
        # hf.create_dataset('box', data=ims)
        # hf.create_dataset('confmaps', data=hmaps)
        # hf.create_dataset('joints', data=locs)
        # hf.create_dataset('exptID', data=eid)
        # hf.create_dataset('framesIdx', data=frid)
        # hf.create_dataset('trxID', data=tid)
        # hf.close()
        #
        # cmd += 'cd {};'.format(dlc_orig_dir)
        # cmd += 'python ' + main_script
        # cmd += ' {}'.format(out_file)
        # cmd += ' -base_output_path {}'.format(conf.cachedir)
        # cmd += ' -epochs {}'.format(int(conf.dl_steps//conf.save_step))
        # cmd += ' -batches_per_epoch {}'.format(conf.save_step)
        # print(cmd)
        # with os.fdopen(tfile,'w') as tmp:
        #     tmp.write(cmd)
        # print(fname)
        # os.system('bash {}'.format(fname))

    def get_pred_fn(self,model_file):
        conf = self.conf
        return leap.training.get_pred_fn(conf, model_file,name='deepnet')

