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



def run_training() :
    this_script_path = os.path.realpath(__file__)
    script_folder_path = os.path.dirname(this_script_path)
    project_folder_path = os.path.dirname(os.path.dirname(script_folder_path))
    # e.g. /groups/branson/bransonlab/taylora/apt

    input_folder_path = os.path.join(project_folder_path, "coco-from-apt/input-folder-read-only")
    output_folder_path = os.path.join(project_folder_path, "coco-from-apt/output-folder-with-db-and-after-aug-pickle")

    # logging.warning('Point 1')

    # Make sure the input folder path exists
    if not os.path.exists(input_folder_path) :
        raise RuntimeError("Read-only test input folder is missing, expected it at %s" % input_folder_path)

    # Prepare the output folder
    logging.info('Preparing the output folder...')
    if os.path.exists(output_folder_path) :
        subprocess.run(['rm', '-rf', output_folder_path], 
                       check=True)
    subprocess.run(['cp', '-R', '-T', input_folder_path, output_folder_path], 
                   check=True)
    logging.info('Done preparing the output folder.')

    # logging.warning('Point 2')

    # Set some CUDA-related envars
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Want the CUDA ID #s for the GPUs to match those used by nvidia-smi and nvtop
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # logging.warning('Point 3')

    datetime_1 = '20231208T165249'
    datetime_2 = '20231208T165249'
    training_subfolder_name = 'coco'
    with cd(script_folder_path):
        APT_interface.main(
            [os.path.join(output_folder_path, training_subfolder_name, f'{datetime_1}_{datetime_2}.json'),
             '-name', datetime_1, 
             '-err_file', os.path.join(output_folder_path,
                                       training_subfolder_name,
                                       f'{datetime_1}view0_{datetime_2}_bu.err'),
             '-json_trn_file', os.path.join(output_folder_path, training_subfolder_name, 'loc.json'), 
             '-conf_params',
             '-type', 'cid',
             '-ignore_local', '0', 
             '-cache', output_folder_path,
             '-no_except',
             '-zero_seeds',
             '-debug',  # Turn on debug-level logging, and loading of data in same process as training, and gradient error checking
#             '-img_prefix_override', '/groups/branson/bransonlab/taylora/apt/mmpose-0.29-native/data/coco/train2017',
             'train', 
             '-only_aug',
             '-do_save_after_aug_pickle',
             '-use_cache'])



# For calling from command line
if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    logging.info('About to call run_training()')
    run_training()
    logging.info('Finished run_training()')
