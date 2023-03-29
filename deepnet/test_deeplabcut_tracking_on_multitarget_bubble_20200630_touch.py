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



def test_deeplabcut_tracking_on_multitarget_bubble_20200630_touch() :
    this_script_path = os.path.realpath(__file__)
    script_folder_path = os.path.dirname(this_script_path)
    project_folder_path = os.path.dirname(os.path.dirname(script_folder_path))  # e.g. /groups/branson/bransonlab/taylora/apt-refactoring

    read_only_folder_path = os.path.join(project_folder_path, "multitarget_bubble_20200630_touch_deeplabcut_tracking_test_input_folder_read_only")
    working_folder_path = os.path.join(project_folder_path, "multitarget_bubble_20200630_touch_deeplabcut_tracking_test_working_folder")

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
            [os.path.join(working_folder_path,'multitarget_bubble/20230327T203619_20230327T203619.json'),
             '-name', '20230327T203619', 
             '-err_file', os.path.join(working_folder_path, 'multitarget_bubble/deeplabcut/view_0/20230327T203619/trk/track_20230327T210247_mov1_vw1.err'),
             '-type', 'deeplabcut',
             '-model_files', os.path.join(working_folder_path, "multitarget_bubble/deeplabcut/view_0/20230327T203619/deepnet-67000"),
             '-ignore_local', '0', 
             '-cache', working_folder_path,
             'track',
             '-config_file', os.path.join(working_folder_path, "multitarget_bubble/deeplabcut/view_0/20230327T203619/trk/trkconfig_movie_trn20230327T203619_view0_iter67000_20230327T210247.json"),
             '-out', os.path.join(working_folder_path, "multitarget_bubble/deeplabcut/view_0/20230327T203619/trk/movie_trn20230327T203619_view0_iter67000_20230327T210247.trk"),
             '-mov', "/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00238_CsChr_RigC_20151007T150343/movie.ufmf",
             '-start_frame', '16271',
             '-end_frame', '16471',
             '-trx', "/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00238_CsChr_RigC_20151007T150343/registered_trx.mat",
             '-trx_ids', '1', '3', '4', '5', '6', '7', '8', '9', '10'])


# python \
#   "/groups/branson/bransonlab/taylora/apt-refactoring/apt-ampere/deepnet/APT_interface.py" \
#   "/groups/scicompsoft/home/taylora/tp5e6097bb_e699_43df_89d5_66367c680eb0/multitarget_bubble/20230327T203619_20230327T203619.json" \
#   -name 20230327T203619 \
#   -err_file "/groups/scicompsoft/home/taylora/tp5e6097bb_e699_43df_89d5_66367c680eb0/multitarget_bubble/deeplabcut/view_0/20230327T203619/trk/track_20230327T210247_mov1_vw1.err" \
#   -type deeplabcut \
#   -model_files "/groups/scicompsoft/home/taylora/tp5e6097bb_e699_43df_89d5_66367c680eb0/multitarget_bubble/deeplabcut/view_0/20230327T203619/deepnet-67000" \
#   -ignore_local 0 \
#   -cache "/groups/scicompsoft/home/taylora/tp5e6097bb_e699_43df_89d5_66367c680eb0" \
#   track \
#   -config_file "/groups/scicompsoft/home/taylora/tp5e6097bb_e699_43df_89d5_66367c680eb0/multitarget_bubble/deeplabcut/view_0/20230327T203619/trk/trkconfig_movie_trn20230327T203619_view0_iter67000_20230327T210247.json" \
#   -out "/groups/scicompsoft/home/taylora/tp5e6097bb_e699_43df_89d5_66367c680eb0/multitarget_bubble/deeplabcut/view_0/20230327T203619/trk/movie_trn20230327T203619_view0_iter67000_20230327T210247.trk" \
#   -mov "/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00238_CsChr_RigC_20151007T150343/movie.ufmf" \
#   -start_frame 16271 \
#   -end_frame 16471 \
#   -trx "/groups/branson/home/robiea/Projects_data/Labeler_APT/cx_GMR_SS00238_CsChr_RigC_20151007T150343/registered_trx.mat" \
#   -trx_ids 1   3   4   5   6   7   8   9  10



# For calling from command line
if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    logging.info('About to call test_deeplabcut_tracking_on_multitarget_bubble_20200630_touch()')
    test_deeplabcut_tracking_on_multitarget_bubble_20200630_touch()
    logging.info('Finished test_deeplabcut_tracking_on_multitarget_bubble_20200630_touch()')
