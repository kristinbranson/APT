#! /usr/bin/env python3

import os
import subprocess
import logging
import APT_interface



class cd :
    """Context manager for changing the current working directory, and automagically changing back when done"""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.old_path = os.getcwd()
        os.chdir(self.path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.old_path)



def test_whatever() :
    this_script_path = os.path.realpath(__file__)
    script_folder_path = os.path.dirname(this_script_path)
    project_folder_path = os.path.dirname(os.path.dirname(script_folder_path))  # e.g. /groups/branson/bransonlab/taylora/apt-refactoring

    read_only_folder_path = os.path.join("/groups/branson/bransonlab/taylora/apt/carmen-repro-read-only")
    working_folder_path = os.path.join("/groups/branson/bransonlab/taylora/apt/carmen-repro")

    #logging.warning('Point 1')

    # Make sure the read-only test folder path exists
    if not os.path.exists(read_only_folder_path) :
        raise RuntimeError("Read-only test input folder is missing, expected it at %s" % read_only_folder_path)

    # Prepare the working folder
    if os.path.exists(working_folder_path) :
        subprocess.run(['rm', '-rf', working_folder_path], 
                       check=True)
    subprocess.run(['cp', '-R', '-T', read_only_folder_path, working_folder_path], 
                   check=True)

    #logging.warning('Point 2')

    # Set some CUDA-related envars
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Want the CUDA ID #s for the GPUs to match those used by nvidia-smi and nvtop
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    #logging.warning('Point 3')

    #deepnet_folder_path = os.path.join(script_folder_path, 'APT/deepnet') 
    #  '-err_file', os.path.join(working_folder_path, 'test1/20241024T004246view0_20241024T004246_sa.err'),
    #  '-log_file', os.path.join(working_folder_path, 'test1/20241024T004246view0_20241024T004246_sa_new.log'),
    deepnet_folder_path = script_folder_path
    with cd(deepnet_folder_path) :
        APT_interface.main(
            [os.path.join(working_folder_path,'test1/20241024T004246_20241024T004246.json'),
             '-name', '20241024T004246', 
             '-json_trn_file', os.path.join(working_folder_path, 'test1/loc.json'), 
             '-conf_params', 'dl_steps', '1000', 
             '-type', 'mdn_joint_fpn',
             '-ignore_local', '0', 
             '-cache', working_folder_path,
             'train', 
             '-use_cache'])


# python \
# "/groups/branson/bransonlab/taylora/apt-refactoring/apt-ampere/deepnet/APT_interface.py" \
# "/groups/scicompsoft/home/taylora/tp0ceda4d9_8b79_4793_a90e_5634a03d3e4a/multitarget_bubble/20230324T175649_20230324T175649.json" \
# -name 20230324T175649 \
# -err_file "/groups/scicompsoft/home/taylora/tp0ceda4d9_8b79_4793_a90e_5634a03d3e4a/multitarget_bubble/20230324T175649view0_20230324T175649_tdptrx.err" \
# -json_trn_file "/groups/scicompsoft/home/taylora/tp0ceda4d9_8b79_4793_a90e_5634a03d3e4a/multitarget_bubble/loc.json" \
# -conf_params \
# -type mdn_joint_fpn \
# -ignore_local 0 \
# -cache "/groups/scicompsoft/home/taylora/tp0ceda4d9_8b79_4793_a90e_5634a03d3e4a" \
# train \
# -use_cache



# For calling from command line
if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    logging.info('About to call test_whatever()')
    test_whatever()
    logging.info('Finished test_whatever()')
