#! /usr/bin/env python3

import os
import subprocess
import logging
#import APT_interface
import pickle
import numpy as np
import torch
from argparse import Namespace
import random
import time
import mmcv
import mmpose.utils



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

    read_only_folder_path = os.path.join(project_folder_path, "coco/coco-input-folder-ready-to-train-with-tf-replaced-read-only")
    working_folder_path = os.path.join(project_folder_path, "coco/coco-working-folder-ready-to-train-with-tf-replaced")

    # Make sure the read-only test folder path exists
    if not os.path.exists(read_only_folder_path) :
        raise RuntimeError("Read-only test input folder is missing, expected it at %s" % read_only_folder_path)

    # # Prepare the working folder
    # logging.info('Preparing the working folder...')
    # if os.path.exists(working_folder_path) :
    #     subprocess.run(['rm', '-rf', working_folder_path], 
    #                    check=True)
    # subprocess.run(['cp', '-R', '-T', read_only_folder_path, working_folder_path], 
    #                check=True)
    # logging.info('Done preparing the working folder.')

    # Set some CUDA-related envars
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Want the CUDA ID #s for the GPUs to match those used by nvidia-smi and nvtop
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Run the training
    with cd(script_folder_path):
        # Load the pickle
        pickle_file_path = os.path.join(project_folder_path, 'coco', 'just-before-train-wrapper.pkl')
        with open(pickle_file_path, 'rb') as f:
            d = pickle.load(f)
        net_type = d['net_type']
        raw_args = d['args']
        restore = d['restore']
        model_file = d['model_file']
        conf = d['conf']

        # Replace the working directory from when these things were created with our current working directory
        original_working_folder_path = '/groups/branson/bransonlab/taylora/apt/coco/coco-working-folder-with-db'
        # For args
        raw_args_as_dict = vars(raw_args)
        args_as_dict = { key: (value.replace(original_working_folder_path, working_folder_path) if isinstance(value, str) else value) 
                         for key, value in raw_args_as_dict.items() }
        args = Namespace(**args_as_dict)
        # For conf (done in-place)
        conf.json_trn_file = conf.json_trn_file.replace(original_working_folder_path, working_folder_path)
        conf.labelfile = conf.labelfile.replace(original_working_folder_path, working_folder_path)
        conf.cachedir = conf.cachedir.replace(original_working_folder_path, working_folder_path)

        # Set up mmpose-style logging, just to make debugging easier
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file_path = os.path.join(working_folder_path, f'{timestamp}.log')
        mmpose_logger = mmpose.utils.get_root_logger(log_file=log_file_path, log_level='INFO')
        # Note that this is the root logger for MMPose, not the 'true' root logger you'd 
        # get by calling logging.getLogger()

        # Run the meaty part of APT_interface::train()
        if net_type == 'mmpose' or net_type == 'hrformer':
            module_name = 'Pose_mmpose'
        elif net_type == 'cid':
            module_name = 'Pose_multi_mmpose'
        else :
            module_name = 'Pose_{}'.format(net_type)                    
        logging.info(f'Importing pose module {module_name}')
        pose_module = __import__(module_name)
        poser_factory = getattr(pose_module, module_name)
        poser = poser_factory(conf, name=args.train_name)
        poser.cfg.data.train['img_prefix'] = '/groups/branson/bransonlab/taylora/apt/mmpose-0.29-native/data/coco/train2017'
        if args.zero_seeds:
            # Set a bunch of seeds to zero for training reproducibility
            seed = 0
            poser.cfg.seed = seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            # Do what we can to "determinize" mmpose.  May be redundant.
            mmcv.runner.set_random_seed(seed, deterministic=True)

            #os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # Needed for cublas determinism
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Needed for cublas determinism
            torch.use_deterministic_algorithms(True, warn_only=True)
        # Proceed to actual training
        logging.info('Starting training...')
        poser.train_wrapper(restore=restore, model_file=model_file, debug=False, logger=mmpose_logger)
        logging.info('Finished training.')



# For calling from command line
if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    logging.info('About to call test_training()')
    test_training()
    logging.info('Finished test_training()')
