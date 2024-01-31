#! /usr/bin/python3

# Designed to work with Python >=3.6, using only the standard library

import sys
import os
import argparse
import subprocess



# These are argument keys that are either a file name or a list of file names.    
file_name_arg_keys = ['lbl_file', 'mov', 'trx', 'log_file', 'err_file', 'out_files']



def apt_track_parse_args(argv):
    # This is copied-and-pasted from APT_track.py.
    # include'ing it doesn't work b/c APT_track.py includes e.g. tensorflow, which
    # we don't want to depend on.  So this may have to change when APT_track.py changes.
    #
    # You should leave this as-is, so that if APT_track.py::parge_args() changes, it can just 
    # be copied-and-pasted again to bring it in sync.

    parser = argparse.ArgumentParser(description='Track movies using the latest trained model in the APT lbl file')
    parser.add_argument("-lbl_file", help="path to APT lbl file or a directory when the lbl file has been unbundled using untar")
#    parser.add_argument('-backend',help='Backend to use for tracking. Options are docker and conda',default='docker')
    parser.add_argument('-list', help='Lists all the trained models present in the lbl file',action='store_true')
    parser.add_argument("-model_ndx", help="Use this model number (model numbers can be found using -list) instead of the latest model",type=int,default=None)
    parser.add_argument("-mov", dest="mov",help="movie(s) to track. For multi-view projects, specify movies for all the views or specify the view for the single movie using -view", nargs='+')
    parser.add_argument("-trx", dest="trx",help='trx file for movie', default=None, nargs='*')
    parser.add_argument("-log_file", dest="log_file",help='log file for output', default=None)
    parser.add_argument("-err_file", dest="err_file",help='error file for output', default=None)
    parser.add_argument('-start_frame', dest='start_frame', help='start frame for tracking', nargs='*', type=int, default=1)
    parser.add_argument('-end_frame', dest='end_frame', help='end frame for tracking', nargs='*', type=int, default=-1)
    parser.add_argument('-out', dest='out_files', help='file to save tracking results to. If track_type is "predict_track" and no predict_trk_files are specified, the pure linked tracklets will be saved to files with _pure suffix. If track_type is "predict_only" the out file will have the pure linked tracklets and predict_trk_files will be ignored.', nargs='+')
    parser.add_argument('-crop_loc', dest='crop_loc', help='crop locations given as xlo xhi ylo yhi', nargs='*', type=int, default=None)
    parser.add_argument('-view', help='track only for this view. If not specified, track for all the views', default=None, type=int)
    parser.add_argument('-stage', help='Stage for multi-stage tracking. Options are multi, first, second or None (default)', default=None)

    parser.add_argument('-track_type',choices=['predict_link','only_predict','only_link'], default='predict_link', help='for multi-animal. Whether to link the predictions or not, or only link existing tracklets. "predict_link" both predicts and links, "only_predict" only predicts but does not link, "only_link" only links existing predictions. For only_link, trk files with raw unlinked predictions must be supplied using -predict_trk_files option.')
    parser.add_argument('-predict_trk_files', help='Intermediate trk files storing pure tracklets. Required when using link_only track_type', default=None, type=int)

    parser.add_argument('-conf_params',
                        help='conf params. These will override params from lbl file. If the model is a 2 stage tracker then this will override the params only for the first stage', default=None, nargs='*')
    parser.add_argument('-conf_params2',
                        help='conf params for 2nd stage. These will override params from lbl file for 2nd stage tracker if the model is a 2 stage tracker', default=None, nargs='*')
    
    args = parser.parse_args(argv)
    return args



def parse_args(argv):
    # Parse all the args.  Split them into the args for us versus the args for APT_track.py.
    main_parser = argparse.ArgumentParser(description='Track movies using the latest trained model in the APT lbl file')
    main_parser.add_argument('-backend', help='Backend to use for tracking. Options are conda and docker', default='conda')
    args_as_namespace, apt_track_tokens = main_parser.parse_known_args(argv)
    args = vars(args_as_namespace)  # want a dict

    # We also run a "side parser" to extract some of the args to APT_track.py, so that we can make sure those are available  
    # in, e.g., a Docker container
    apt_track_args_as_namespace = apt_track_parse_args(apt_track_tokens)
    apt_track_args = vars(apt_track_args_as_namespace)  # Want a dict

    return args, apt_track_args



def flatten(lst):
    # Flatten a list of lists.
    result = []
    for item in lst:
        if isinstance(item, list):
            result += flatten(item)
        else:
            result.append(item)
    return result



def mount_string_from_file_name(file_name):
    # Generate a string that will work in a docker --mount argument.
    # We assume the source and target have the same path.
    # We call such a string a "mount string"
    file_absolute_path = os.path.realpath(file_name)
    result = 'type=bind,source=%s,target=%s' % (file_absolute_path, file_absolute_path)
    return result



def mount_string_lists_from_apt_arg_dict_pair(name, value):
    # Given a single (name, value) pair from the suppied arguments to APT_track, 
    # Generate a list of mount strings that will be used to make sure docker can 
    # access the needed files.
    if name in file_name_arg_keys:
        if isinstance(value, list):
            result = [ mount_string_from_file_name(file_name) for file_name in value ]
        elif value is None:
            result = []
        else:
            file_name = value
            result = [ mount_string_from_file_name(file_name) ]
    else:
        result = []
    return result



def mount_tokens_from_apt_track_args(apt_track_args):
    # Given a dict of APT_Track.pt arguments, outputs a list of command-line tokens to be used when
    # calling docker run, to ensure that all the needed files are available in the container.
    # We call this a list of "mount tokens".
    raw_mount_strings = [ mount_string_lists_from_apt_arg_dict_pair(name, value) for name,value in apt_track_args.items() ]
    mount_strings = flatten(raw_mount_strings)
    mount_tokens_as_list_of_lists = [ ['--mount', mount_string] for mount_string in mount_strings ]
    mount_tokens = flatten(mount_tokens_as_list_of_lists)
    return mount_tokens



def tracklet_mount_tokens_from_output_file_name(output_file_name):
    # Returns a list of mount tokens that will enable the docker container to write to the "tracklet" file.
    # If the output file name is e.g. out.trk, the tracklet file is named out_tracklet.trk.
    output_file_absolute_path = os.path.realpath(output_file_name)
    #print('output_file_absolute_path:')
    #print(output_file_absolute_path)
    [output_file_folder_path, output_file_leaf_name] = os.path.split(output_file_absolute_path)
    [output_file_base_name, output_file_extension] = os.path.splitext(output_file_leaf_name)
    tracklet_file_base_name = output_file_base_name + '_tracklet'
    tracklet_file_leaf_name = tracklet_file_base_name + output_file_extension
    tracklet_file_absolute_path = os.path.join(output_file_folder_path, tracklet_file_leaf_name)
    tracklet_mount_tokens = [ '--mount', mount_string_from_file_name(tracklet_file_absolute_path) ]
    return tracklet_mount_tokens



def tracklet_mount_tokens_from_output_file_names(output_file_names):
    # Like tracklet_mount_tokens_from_output_file_name(), but works on a list of output file names
    unflattened_result = [ tracklet_mount_tokens_from_output_file_name(output_file_name) for output_file_name in output_file_names ]
    result = flatten(unflattened_result) 
    return result



def reify_paths_in_apt_arg_dict(apt_arg_dict):
    # Convert all the file/dir names in apt_arg_dict to
    # os.path.realpath()'ed versions.  Note this does not modify the input.
    result = {}
    for key,value in apt_arg_dict.items():        
        if key in file_name_arg_keys:
            if isinstance(value, list):
                new_value = [ os.path.realpath(file_name) for file_name in value ]
            elif value is None:
                new_value = None
            else:
                new_value = os.path.realpath(value)        
        else:
            new_value = value
        result[key] = new_value
    return result



def stringify(x):
    # Given an object, which might be a None, a string, an int, or a (possibly recursive) list of one of these,
    # convert to a (possibly recursive) list of strings, which preserves the original list structure.
    if isinstance(x, list):
        result = [ stringify(el) for el in x ]
    else:
        result = str(x)
    return result



def apt_track_tokens_from_apt_track_args_helper(key, value):
    if value is None:
        result = []
    elif isinstance(value, bool):
        if value:
            result = [ '-' + key ]
        else:
            result = []
    else:
        value_as_token_or_token_list = stringify(value)
        if isinstance(value_as_token_or_token_list, str):
            value_as_token_list = [ value_as_token_or_token_list ]
        else:
            value_as_token_list = value_as_token_or_token_list
        if key=='out_files':
            rekey = '-out'
        else:
            rekey = '-' + key
        result = [rekey] + value_as_token_list        

    print('')
    print('key:')
    print(key)
    print('value:')
    print(value)
    print('result:')
    print(result)

    return result



def apt_track_tokens_from_apt_track_args(apt_track_args):
    protoresult = [ apt_track_tokens_from_apt_track_args_helper(key, value) for key, value in apt_track_args.items() ]
    result = flatten(protoresult)
    return result



def run_with_conda(apt_track_py_absolute_path, apt_track_args, apt_track_tokens):
    environment_name = 'apt_20230427_tf211_pytorch113_ampere'
    command_as_list = ['conda', 'run', '--live-stream', '-n', environment_name, 'python', apt_track_py_absolute_path] + apt_track_tokens
    subprocess.run(command_as_list, check=True)



def run_with_docker(apt_track_py_absolute_path, apt_track_args, apt_track_tokens):
    # Get the '--mount' command-line tokens from the APT_track arguments 
    apt_track_mount_tokens = mount_tokens_from_apt_track_args(apt_track_args)

    # Specify the tag of the image to use
    image_tag = 'bransonlabapt/apt_docker:apt_20230427_tf211_pytorch113_ampere'

    # Get the '--mount' command-line tokens for the APT deepnet/ folder
    deepnet_folder_absolute_path = os.path.dirname(apt_track_py_absolute_path)
    deepnet_mount_tokens = [ '--mount', mount_string_from_file_name(deepnet_folder_absolute_path) ]

    # Add mount items for the tracklet file
    if 'out_files' in apt_track_args.keys():
        output_file_names = apt_track_args['out_files']
        tracklet_mount_tokens = tracklet_mount_tokens_from_output_file_names(output_file_names)
    else:
        # What to do here?
        tracklet_mount_tokens = []

    # Assemble the command line (as a list of tokens)
    command_as_list = [ 'docker', 'run', '--gpus', 'all' ] + \
                      deepnet_mount_tokens + \
                      apt_track_mount_tokens + \
                      tracklet_mount_tokens + \
                      [ image_tag, 'python', apt_track_py_absolute_path ] + \
                      apt_track_tokens
    print('command_as_list:')
    print(command_as_list)                  

    # Finally, execute the command line
    subprocess.run(command_as_list, check=True)



def main(argv):
    # Get an absolute path to APT_track.py
    this_script_path = os.path.realpath(__file__)
    this_script_folder_path = os.path.dirname(this_script_path)
    apt_track_py_absolute_path = os.path.join(this_script_folder_path, 'APT_track.py')

    (args, apt_track_args) = parse_args(argv)
       # args is a namespace, leftover_tokens is a list of strings
    print('args:')
    print(args) 
    print('apt_track_args:')
    print(apt_track_args)

    # Convert the paths in apt_track_args to realpath'ed paths
    massaged_apt_track_args = reify_paths_in_apt_arg_dict(apt_track_args)
    print('massaged_apt_track_args:')
    print(massaged_apt_track_args)

    # Remake the tokens to be used in the call to APT_track.py, to use the reified paths
    apt_track_tokens = apt_track_tokens_from_apt_track_args(massaged_apt_track_args)
    print('apt_track_tokens:')
    print(apt_track_tokens)

    backend = args['backend']
    if backend=='conda':
        run_with_conda(apt_track_py_absolute_path, massaged_apt_track_args, apt_track_tokens)
    if backend=='docker':
        run_with_docker(apt_track_py_absolute_path, massaged_apt_track_args, apt_track_tokens)
    else:
        raise RuntimeError('%s is not a valid backend.  Valid options are conda, and docker.' % args.backend)



if __name__ == "__main__":
    main(sys.argv[1:])
