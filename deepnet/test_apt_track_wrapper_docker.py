import os
import sys
import importlib

# Designed to work with Python >=3.6, using only the standard library

# Get path this this folder
SCRIPT_FILE_PATH = os.path.realpath(__file__)
SCRIPT_FOLDER_PATH = os.path.dirname(SCRIPT_FILE_PATH)
apt_track_wrapper_path = os.path.join(SCRIPT_FOLDER_PATH, 'apt-track-wrapper')

# Import apt-track-wrapper, which is good and complicated!
module_name = 'apt_track_wrapper'
spec = importlib.util.spec_from_loader(module_name, importlib.machinery.SourceFileLoader(module_name, apt_track_wrapper_path))
if spec is None:
    raise ImportError(f"Could not load spec for module '{module_name}' at: {apt_track_wrapper_path}")
apt_track_wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apt_track_wrapper)
sys.modules[module_name] = apt_track_wrapper

# Define input/output file paths
TEST_SCRIPT_PATH = os.path.join(SCRIPT_FOLDER_PATH, 'apt-track')
LABEL_FILE_PATH = '/groups/branson/bransonlab/taylora/apt/four-points/four-points-testing-2024-02-07-track-wrapper-branch.lbl'
VIDEO_FILE_PATH = '/groups/branson/bransonlab/roian/apt_testing/files_for_working_with_apt/190530_vocpbm164564_m164564_odor_m164301_f0_164992.mjpg'
OUTPUT_FILE_PATH = './apt-track-test-from-script.trk'

apt_track_wrapper.main(['-docker', 'bransonlabapt/apt_docker:apt_20230427_tf211_pytorch113_ampere', '-lbl_file', LABEL_FILE_PATH, '-mov', VIDEO_FILE_PATH, '-start_frame', '1', '-end_frame', '200', '-out', OUTPUT_FILE_PATH])
