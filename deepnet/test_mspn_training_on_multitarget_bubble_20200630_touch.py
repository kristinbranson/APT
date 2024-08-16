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



def test_mspn_training_on_multitarget_bubble_20200630_touch() :
    this_script_path = os.path.realpath(__file__)
    script_folder_path = os.path.dirname(this_script_path)
    project_folder_path = os.path.dirname(os.path.dirname(script_folder_path))  # e.g. /groups/branson/bransonlab/taylora/apt-refactoring

    read_only_folder_path = os.path.join(project_folder_path, "multitarget_bubble_20200630_touch_mspn_training_test_input_folder_read_only")
    working_folder_path = os.path.join(project_folder_path, "multitarget_bubble_20200630_touch_mspn_training_test_working_folder")

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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    #logging.warning('Point 3')

    #deepnet_folder_path = os.path.join(script_folder_path, 'APT/deepnet') 
    deepnet_folder_path = script_folder_path
    with cd(deepnet_folder_path) :
        APT_interface.main(
            ['-name', '20221204T230902', 
             '-view', '1', 
             '-cache', working_folder_path,
             '-err_file', os.path.join(working_folder_path, 'multitarget_bubble/20221204T230902view0_20221204T231014_tdptrx.err'),
             '-json_trn_file', os.path.join(working_folder_path, 'multitarget_bubble/loc.json'), 
             '-conf_params', 'dl_steps', '200',  # 1000 takes too long
             '-type', 'mmpose',
             os.path.join(working_folder_path,'multitarget_bubble/20221204T230902_20221204T231014.json'),
             'train', 
             '-use_cache'])



# For calling from command line
if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    logging.info('About to call test_mspn_training_on_multitarget_bubble_20200630_touch()')
    test_mspn_training_on_multitarget_bubble_20200630_touch()
    logging.info('Finished test_mspn_training_on_multitarget_bubble_20200630_touch()')
