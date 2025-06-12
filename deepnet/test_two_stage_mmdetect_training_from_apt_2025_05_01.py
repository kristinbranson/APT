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



def test_training() :
    this_script_path = os.path.realpath(__file__)
    deepnet_folder_path = os.path.dirname(this_script_path)

    read_only_folder_path = "/groups/branson/bransonlab/taylora/apt/repro-files/two-stage-mmdetect-2025-05-01/test-files-read-only"
    working_folder_path = "/groups/branson/bransonlab/taylora/apt/repro-files/two-stage-mmdetect-2025-05-01/test-files-working"

    # Make sure the read-only test folder path exists
    if not os.path.exists(read_only_folder_path) :
        raise RuntimeError("Read-only test input folder is missing, expected it at %s" % read_only_folder_path)

    # Prepare the working folder
    if os.path.exists(working_folder_path) :
        subprocess.run(['rm', '-rf', working_folder_path], 
                       check=True)
    subprocess.run(['cp', '-R', '-T', read_only_folder_path, working_folder_path], 
                   check=True)

    # Set some CUDA-related envars
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Want the CUDA ID #s for the GPUs to match those used by nvidia-smi and nvtop
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    with cd(deepnet_folder_path):
        APT_interface.main(
            [os.path.join(working_folder_path,'fourpoints512/20250501T110316_20250501T110316.json'),
             '-name', '20250501T110316', 
             '-json_trn_file', os.path.join(working_folder_path, 'fourpoints512/loc.json'), 
             '-stage', 'multi',
             '-conf_params',
             '-type', 'detect_mmdetect',
             '-conf_params2', 'use_bbox_trx', 'True',
             '-type2', 'mdn_joint_fpn',
             '-ignore_local', '0', 
             '-cache', working_folder_path,
             'train', 
             '-use_cache'])



# For calling from command line
if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    logging.info('About to call test_training()')
    test_training()
    logging.info('Finished test_training()')
