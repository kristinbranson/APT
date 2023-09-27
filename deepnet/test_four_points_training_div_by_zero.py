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
    script_folder_path = os.path.dirname(this_script_path)
    project_folder_path = os.path.dirname(os.path.dirname(script_folder_path))
    # e.g. /groups/branson/bransonlab/taylora/apt

    read_only_folder_path = os.path.join(project_folder_path, "four_points_div_by_zero_tp_input_folder_read_only")
    working_folder_path = os.path.join(project_folder_path, "four_points_div_by_zero_tp_working_folder")

    # logging.warning('Point 1')

    # Make sure the read-only test folder path exists
    if not os.path.exists(read_only_folder_path) :
        raise RuntimeError("Read-only test input folder is missing, expected it at %s" % read_only_folder_path)

    # Prepare the working folder
    if os.path.exists(working_folder_path) :
        subprocess.run(['rm', '-rf', working_folder_path], 
                       check=True)
    subprocess.run(['cp', '-R', '-T', read_only_folder_path, working_folder_path], 
                   check=True)

    # logging.warning('Point 2')

    # Set some CUDA-related envars
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Want the CUDA ID #s for the GPUs to match those used by nvidia-smi and nvtop
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # logging.warning('Point 3')

    deepnet_folder_path = script_folder_path
    with cd(deepnet_folder_path):
        # APT_interface.IS_APT_IN_DEBUG_MODE = True
        APT_interface.main(
            [os.path.join(working_folder_path,'four_points_180806/20230830T155705_20230830T155706.json'),
             '-name', '20230830T155705', 
             '-err_file', os.path.join(working_folder_path,
                                       'four_points_180806/20230830T155705view0_20230830T155706_bu.err'),
             '-json_trn_file', os.path.join(working_folder_path, 'four_points_180806/loc.json'), 
             '-conf_params', 'dl_steps', '200',
             '-type', 'cid',
             '-ignore_local', '0', 
             '-cache', working_folder_path,
             '-debug',
             'train', 
             '-use_cache'])



# For calling from command line
if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    logging.info('About to call test_training()')
    test_training()
    logging.info('Finished test_training()')