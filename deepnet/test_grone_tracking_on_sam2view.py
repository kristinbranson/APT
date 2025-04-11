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



def test_grone_tracking_on_multitarget_bubble_20200630_touch() :
    this_script_path = os.path.realpath(__file__)
    script_folder_path = os.path.dirname(this_script_path)
    project_folder_path = os.path.dirname(os.path.dirname(script_folder_path))  # e.g. /groups/branson/bransonlab/taylora/apt-refactoring

    read_only_folder_path = "/groups/branson/bransonlab/taylora/apt/apt-interface-repro-folders/sam2view-repro-read-only"
    working_folder_path = "/groups/branson/bransonlab/taylora/apt/apt-interface-repro-folders/sam2view-repro"

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
    deepnet_folder_path = script_folder_path
    with cd(deepnet_folder_path) :
        APT_interface.main(
            [os.path.join(working_folder_path,'2011_mouse_cam13/20241111T173037_20241111T173037.json'),
             '-name', '20241111T173037', 
             '-err_file', os.path.join(working_folder_path, '2011_mouse_cam13/mdn_joint_fpn/view_0/20241111T173037/trk/track_20241111T180238_mov1_vw1.err'),
             '-type', 'mdn_joint_fpn',
             '-model_files', os.path.join(working_folder_path, "2011_mouse_cam13/mdn_joint_fpn/view_0/20241111T173037/deepnet-1000"),
                             os.path.join(working_folder_path, "2011_mouse_cam13/mdn_joint_fpn/view_0/20241111T173037/deepnet-1000"),
             '-ignore_local', '0', 
             '-cache', working_folder_path,
             'track',
             '-config_file', os.path.join(working_folder_path, "2011_mouse_cam13/mdn_joint_fpn/view_0/20241111T173037/trk/trkconfig_day1_avg1_2021_11_15_16_03_24_0_e59274_trn20241111T173037_view0_iter1000_20241111T180238.json"),
             '-out', os.path.join(working_folder_path, "2011_mouse_cam13/mdn_joint_fpn/view_0/20241111T173037/trk/day1_avg1_2021_11_15_16_03_24_0_e59274_trn20241111T173037_view0_iter1000_20241111T180238.trk"),
                     os.path.join(working_folder_path, "2011_mouse_cam13/mdn_joint_fpn/view_1/20241111T173037/trk/day1_avg1_2021_11_15_16_03_24_2_eb8ee9_trn20241111T173037_view1_iter1000_20241111T180238.trk"),  
             '-mov', "/groups/branson/bransonlab/DataforAPT/JumpingMice/2021_11_adultCtxOpto/day1_avg1_2021_11_15_16_03_24_0.avi",
                     "/groups/branson/bransonlab/DataforAPT/JumpingMice/2021_11_adultCtxOpto/day1_avg1_2021_11_15_16_03_24_2.avi",
             '-start_frame', '5687',
             '-end_frame', '5887',
             '-trx_ids', '1'])



# python /groups/branson/bransonlab/taylora/apt/apt-backup-model-2/deepnet/APT_interface.py \
#     /groups/branson/bransonlab/taylora/.apt/tp0583edd8_75bf_4294_b210_0d42daddf731/2011_mouse_cam13/20241111T173037_20241111T173037.json \
#     -name 20241111T173037 \
#     -err_file /groups/branson/bransonlab/taylora/.apt/tp0583edd8_75bf_4294_b210_0d42daddf731/2011_mouse_cam13/mdn_joint_fpn/view_0/20241111T173037/trk/track_20241111T180238_mov1_vw1.err \
#     -type mdn_joint_fpn \
#     -model_files /groups/branson/bransonlab/taylora/.apt/tp0583edd8_75bf_4294_b210_0d42daddf731/2011_mouse_cam13/mdn_joint_fpn/view_0/20241111T173037/deepnet-1000 \
#                  /groups/branson/bransonlab/taylora/.apt/tp0583edd8_75bf_4294_b210_0d42daddf731/2011_mouse_cam13/mdn_joint_fpn/view_1/20241111T173037/deepnet-1000 \
#     -ignore_local 0 \
#     -cache /groups/branson/bransonlab/taylora/.apt/tp0583edd8_75bf_4294_b210_0d42daddf731 \
#     track \
#     -config_file /groups/branson/bransonlab/taylora/.apt/tp0583edd8_75bf_4294_b210_0d42daddf731/2011_mouse_cam13/mdn_joint_fpn/view_0/20241111T173037/trk/trkconfig_day1_avg1_2021_11_15_16_03_24_0_e59274_trn20241111T173037_view0_iter1000_20241111T180238.json \
#     -out /groups/branson/bransonlab/taylora/.apt/tp0583edd8_75bf_4294_b210_0d42daddf731/2011_mouse_cam13/mdn_joint_fpn/view_0/20241111T173037/trk/day1_avg1_2021_11_15_16_03_24_0_e59274_trn20241111T173037_view0_iter1000_20241111T180238.trk \
#          /groups/branson/bransonlab/taylora/.apt/tp0583edd8_75bf_4294_b210_0d42daddf731/2011_mouse_cam13/mdn_joint_fpn/view_1/20241111T173037/trk/day1_avg1_2021_11_15_16_03_24_2_eb8ee9_trn20241111T173037_view1_iter1000_20241111T180238.trk \
#     -mov /groups/branson/bransonlab/DataforAPT/JumpingMice/2021_11_adultCtxOpto/day1_avg1_2021_11_15_16_03_24_0.avi \
#          /groups/branson/bransonlab/DataforAPT/JumpingMice/2021_11_adultCtxOpto/day1_avg1_2021_11_15_16_03_24_2.avi \
#     -start_frame 5687 \
#     -end_frame 5887 \
#     -trx_ids 1


# For calling from command line
if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    logging.info('About to call test_grone_tracking_on_multitarget_bubble_20200630_touch()')
    test_grone_tracking_on_multitarget_bubble_20200630_touch()
    logging.info('Finished test_grone_tracking_on_multitarget_bubble_20200630_touch()')
