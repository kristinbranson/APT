import os
import apt_track_wrapper

SCRIPT_FILE_PATH = os.path.realpath(__file__)
SCRIPT_FOLDER_PATH = os.path.dirname(SCRIPT_FILE_PATH)

TEST_SCRIPT_PATH = os.path.join(SCRIPT_FOLDER_PATH, 'apt-track')
LABEL_FILE_PATH = '/groups/branson/bransonlab/taylora/apt/four-points/four-points-testing-2024-01-26-multianimal-branch.lbl'
VIDEO_FILE_PATH = '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/190530_vocpbm164564_m164564_odor_m164301_f0_164992.mjpg'
OUTPUT_FILE_PATH = './apt-track-test.trk'

apt_track_wrapper.main(['-backend', 'docker', '-lbl_file', LABEL_FILE_PATH, '-mov', VIDEO_FILE_PATH, '-start_frame', '1', '-end_frame', '200', '-out', OUTPUT_FILE_PATH])
