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
    #project_folder_path = os.path.dirname(os.path.dirname(script_folder_path))  # e.g. /groups/branson/bransonlab/taylora/apt-refactoring

    read_only_folder_path = os.path.join("/groups/branson/bransonlab/taylora/apt/apt-interface-repro-folders/carmen-gt-repro-read-only")
    working_folder_path = os.path.join("/groups/branson/bransonlab/taylora/apt/apt-interface-repro-folders/carmen-gt-repro")

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
            [os.path.join(working_folder_path,'test1/20241101T143557_20241101T143557.json'),
             '-name', '20241101T143557', 
             '-type', 'mdn_joint_fpn',
             '-model_files', os.path.join(working_folder_path, 'test1/mdn_joint_fpn/view_0/20241101T143557/deepnet-1000'), 
             '-ignore_local', '0', 
             '-cache', working_folder_path,
             'track', 
             '-config_file', os.path.join(working_folder_path, 'test1/mdn_joint_fpn/view_0/20241101T143557/trk/trkconfig_run010_pez3002_20190729_expt0204000004301139_vid0015_890796_trn20241101T143557_view0_iter1000_20241104T155906.json'),
             '-list_file', os.path.join(working_folder_path, 'TrackList_20241101T143557_20241104T155906.json'), 
             '-out', os.path.join(working_folder_path, 'preds_20241101T143557_20241104T155906_view1.mat')])



# python /groups/branson/bransonlab/taylora/apt/apt-backup-model-2/deepnet/APT_interface.py 
#        /groups/branson/bransonlab/taylora/.apt/tp796648f6_e701_41d8_afd1_061c111e41db/test1/20241101T143557_20241101T143557.json 
#        -name 20241101T143557 
#        -err_file /groups/branson/bransonlab/taylora/.apt/tp796648f6_e701_41d8_afd1_061c111e41db/test1/mdn_joint_fpn/view_0/20241101T143557/trk/track_20241104T155906_list.err 
#        -type mdn_joint_fpn 
#        -model_files /groups/branson/bransonlab/taylora/.apt/tp796648f6_e701_41d8_afd1_061c111e41db/test1/mdn_joint_fpn/view_0/20241101T143557/deepnet-1000 
#        -ignore_local 0 
#        -cache /groups/branson/bransonlab/taylora/.apt/tp796648f6_e701_41d8_afd1_061c111e41db 
#        track 
#        -config_file /groups/branson/bransonlab/taylora/.apt/tp796648f6_e701_41d8_afd1_061c111e41db/test1/mdn_joint_fpn/view_0/20241101T143557/trk/trkconfig_run010_pez3002_20190729_expt0204000004301139_vid0015_890796_trn20241101T143557_view0_iter1000_20241104T155906.json 
#        -list_file /groups/branson/bransonlab/taylora/.apt/tp796648f6_e701_41d8_afd1_061c111e41db/TrackList_20241101T143557_20241104T155906.json 
#        -out /groups/branson/bransonlab/taylora/.apt/tp796648f6_e701_41d8_afd1_061c111e41db/preds_20241101T143557_20241104T155906_view1.mat'




# For calling from command line
if __name__ == "__main__":
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    logging.info('About to call test_whatever()')
    test_whatever()
    logging.info('Finished test_whatever()')
