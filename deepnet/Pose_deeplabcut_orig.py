import PoseTools as pt
import sys
import os
import deeplabcut
import APT_interface as apt
import tempfile
# Use ~/bransonlab/singularity/pytorch_mmpose.sif

dlc_orig_dir  = '/groups/branson/bransonlab/mayank/apt_expts/deepcut_orig2'
main_script = 'run_train.py'
dlc_model_dir = 'dlc-models/iteration-0/aptMayYay-trainset95shuffle1/train/'
class Pose_deeplabcut_orig(object):

    def __init__(self,conf,name='deepnet'):
        self.name = name
        self.conf = conf

    def train_wrapper(self,restore=False,model_file=None):
        conf = self.conf
        fname = os.path.join(conf.cachedir,'run_dlc_orig_{}.sh'.format(self.name))
        cmd = 'source /groups/branson/home/kabram/anaconda3/etc/profile.d/conda.sh;\n'
        cmd += 'conda activate DLC-GPU;\n'
        cmd += 'export DLClight=True\n'
        cmd += 'cd {};'.format(dlc_orig_dir)
        cmd += 'python ' + main_script
        cmd += ' {}'.format(conf.cachedir)
        cmd += ' {}'.format( conf.n_classes)
        # cmd += ' -maxiters {}'.format(8*conf.dl_steps)
        cmd += ' -saveiters {}'.format(conf.save_step*8)
        cmd += ';\n'
        dlc_models = os.path.join(conf.cachedir,dlc_model_dir,'snapshot-*')
        apt_models = os.path.join(conf.cachedir,dlc_model_dir,'{}-*'.format(self.name))
        rename_cmd = 'for file in {}; do mv "$file" "${{file/snapshot-/{}-}}"; done;\n'.format(dlc_models,self.name)
        cmd += rename_cmd
        cmd += 'cp -p {} {};\n'.format(apt_models, conf.cachedir)
        cmd += 'rm -f {};\n'.format(apt_models)
        print(cmd)
        with open(fname,'w') as tmp:
            tmp.write(cmd)
        print(fname)
        os.system('bash {}'.format(fname))

    def get_pred_fn(self,model_file):
        conf = self.conf
        cfg_dict = apt.create_dlc_cfg_dict(conf,'snapshot')
        return deeplabcut.pose_estimation_tensorflow.get_pred_fn(cfg_dict, model_file)
