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



def test_tracking() :
    this_script_path = os.path.realpath(__file__)
    script_folder_path = os.path.dirname(this_script_path)
    project_folder_path = os.path.dirname(os.path.dirname(script_folder_path))
    # e.g. /groups/branson/bransonlab/taylora/apt

    read_only_folder_path = os.path.join(project_folder_path, "four-points-fresh-start-tracking-input-folder-read-only")
    working_folder_path = os.path.join(project_folder_path, "four-points-fresh-start-tracking-working-folder")

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

    # python 
    # '/groups/branson/bransonlab/taylora/apt/apt-cid/deepnet/APT_interface.py' 
    # '/home/taylora@hhmi.org/.apt/tpc82ee696_aa24_45f5_9a1a_40b6392fb57d/four_points_180806/20231006T183351_20231006T183352.json' 
    # -name 20231006T183351 
    # -err_file '/home/taylora@hhmi.org/.apt/tpc82ee696_aa24_45f5_9a1a_40b6392fb57d/four_points_180806/cid/view_0/20231006T183351/trk/track_20231009T165821_mov1_vw1.err' 
    # -type cid 
    # -model_files '/home/taylora@hhmi.org/.apt/tpc82ee696_aa24_45f5_9a1a_40b6392fb57d/four_points_180806/cid/view_0/20231006T183351/deepnet-20000' 
    # -ignore_local 0 
    # -cache '/home/taylora@hhmi.org/.apt/tpc82ee696_aa24_45f5_9a1a_40b6392fb57d' 
    # track 
    # -config_file '/home/taylora@hhmi.org/.apt/tpc82ee696_aa24_45f5_9a1a_40b6392fb57d/four_points_180806/cid/view_0/20231006T183351/trk/trkconfig_190530_vocpbm164564_m164564_odor_m164301_f0_164992_8c1f69_trn20231006T183351_view0_iter20000_20231009T165821.json' 
    # -track_type only_predict 
    # -out '/home/taylora@hhmi.org/.apt/tpc82ee696_aa24_45f5_9a1a_40b6392fb57d/four_points_180806/cid/view_0/20231006T183351/trk/190530_vocpbm164564_m164564_odor_m164301_f0_164992_8c1f69_trn20231006T183351_view0_iter20000_20231009T165821.trk' 
    # -mov '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/190530_vocpbm164564_m164564_odor_m164301_f0_164992.mjpg' 
    # -start_frame 47901 
    # -end_frame 48101

    deepnet_folder_path = script_folder_path
    with cd(deepnet_folder_path):
        APT_interface.main(
            [os.path.join(working_folder_path,'four_points_180806/20231006T183351_20231006T183352.json'),
             '-name', '20231006T183351', 
             '-err_file', os.path.join(working_folder_path,
                                       'four_points_180806/cid/view_0/20231006T183351/trk/track_20231009T165821_mov1_vw1.err'),
             '-type', 'cid',
             '-model_files', os.path.join(working_folder_path, 'four_points_180806/cid/view_0/20231006T183351/deepnet-20000'), 
             '-ignore_local', '0', 
             '-cache', working_folder_path,
             '-no_except',
             'track', 
             '-config_file', os.path.join(working_folder_path, 'four_points_180806/cid/view_0/20231006T183351/trk/trkconfig_190530_vocpbm164564_m164564_odor_m164301_f0_164992_8c1f69_trn20231006T183351_view0_iter20000_20231009T165821.json'),
             '-track_type', 'only_predict',
             '-out', os.path.join(working_folder_path, 'four_points_180806/cid/view_0/20231006T183351/trk/190530_vocpbm164564_m164564_odor_m164301_f0_164992_8c1f69_trn20231006T183351_view0_iter20000_20231009T165821.trk'), 
             '-mov', '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/190530_vocpbm164564_m164564_odor_m164301_f0_164992.mjpg', 
             '-start_frame', '47901',
             '-end_frame', '48101' ] )



# For calling from command line
if __name__ == "__main__":
    #root_logger = logging.getLogger()
    #root_logger.setLevel(logging.DEBUG)
    logging.info('About to call test_tracking()')
    test_tracking()
    logging.info('Finished test_tracking()')
