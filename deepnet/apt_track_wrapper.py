#! /usr/bin/python3

# Designed to work with Python >=3.6, using only the standard library

import sys
import os
import argparse
import subprocess



# In everything that follows:
#   - a "reified" path is one that has been passed through os.path.realpath()



# Default backends for each backend type
conda_default_environment_name = 'apt_20230427_tf211_pytorch113_ampere'
docker_default_image_tag = 'bransonlabapt/apt_docker:apt_20230427_tf211_pytorch113_ampere'
apptainer_default_image_spec = 'docker://bransonlabapt/apt_docker:apt_20230427_tf211_pytorch113_ampere'

# These are argument keys that are either a file name or a list of file names.    
file_name_arg_keys = ['lbl_file', 'mov', 'trx', 'log_file', 'err_file', 'out_files']



def flatten(lst):
    '''
    Flatten a list of lists.
    '''
    result = []
    for item in lst:
        if isinstance(item, list):
            result += flatten(item)
        else:
            result.append(item)
    return result



def select(lst, pred) :
    '''
    Return a list containing only the elements of lst for which pred is true.
    lst and pred should be lists of the same length.
    '''
    result =[]
    for i in range(len(lst)) :
        if pred[i]:
            result.append(lst[i])
    return result



def stringify(x):
    '''
    Given an object, which might be a None, a string, an int, or a (possibly recursive) list of one of these,
    convert to a (possibly recursive) list of strings, which preserves the original list structure.
    '''
    if isinstance(x, list):
        result = [ stringify(el) for el in x ]
    else:
        result = str(x)
    return result



def apt_track_parse_args(argv):
    '''
    This is copied-and-pasted from APT_track.py. include'ing it doesn't work b/c
    APT_track.py includes e.g. tensorflow, which we don't want to depend on.  So
    this may have to change when APT_track.py changes.

    You should leave this as-is, so that if APT_track.py::parge_args() changes,
    it can just be copied-and-pasted again to bring it in sync.
    '''

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
    '''
    Parse all the args.  Split them into the args for us versus the args for
    APT_track.py.
    '''
    main_parser = argparse.ArgumentParser(description='Track movies using the latest trained model in the APT lbl file')
    #main_parser.add_argument('-backend', help='Backend to use for tracking. Options are conda and docker', default='conda')
    main_parser.add_argument('-conda', help='Use conda backend.  An environment name is optional.', nargs='?', default=None, const=conda_default_environment_name)
    main_parser.add_argument('-docker', help='Use docker backend.  An image tag is optional.', nargs='?', default=None, const=docker_default_image_tag)
    main_parser.add_argument('-apptainer', help='Use apptainer backend.  An image spec is optional.', nargs='?', default=None, const=apptainer_default_image_spec)
    args_as_namespace, apt_track_tokens = main_parser.parse_known_args(argv)
    args = vars(args_as_namespace)  # want a dict

    # We also run a "side parser" to extract some of the args to APT_track.py, so that we can make sure those are available  
    # in, e.g., a Docker container
    apt_track_args_as_namespace = apt_track_parse_args(apt_track_tokens)
    apt_track_args = vars(apt_track_args_as_namespace)  # Want a dict

    return args, apt_track_args



def containerize_path(path, container_mount_root_path):
    '''
    Convert an absolute path to the corresponding path in the container.  We
    assume the input path is absolute and reified.
    '''
    if container_mount_root_path is None:
        result = path
    else:
        relativized_path = path[1:]
        result = os.path.join(container_mount_root_path, relativized_path)
    return result


# def mount_string_from_file_name(file_name):
#     '''
#     Generate a string that will work in a docker --mount argument. We assume the
#     source and target have the same path. We call such a string a "mount string"
#     '''
#     file_absolute_path = os.path.realpath(file_name)
#     result = 'type=bind,source=%s,target=%s' % (file_absolute_path, file_absolute_path)
#     return result



def mount_string_from_folder_path(folder_path, container_mount_root_path):
    '''
    Generate a string that will work in a docker --mount argument.
    folder_path is assumed to be absolute, reified, and normed.
    We call such a string a "mount string"
    '''
    container_folder_path = containerize_path(folder_path, container_mount_root_path)
    result = 'type=bind,source=%s,target=%s' % (folder_path, container_folder_path)
    return result



def mount_folder_list_from_reified_apt_arg_dict_pair(key, value):
    '''
    Given a single (name, value) pair from the suppied arguments to APT_track, 
    that has already gone through path reification, generate a list of folders 
    that should be mounted into the container, so that docker can access them.
    '''
    if key in file_name_arg_keys:
        if isinstance(value, list):
            result = [ os.path.normpath(os.path.dirname(file_absolute_path)) for file_absolute_path in value ]
        elif value is None:
            result = []
        else:
            file_absolute_path = value
            result = [ os.path.normpath(os.path.dirname(file_absolute_path)) ]
    else:
        result = []
    return result



def prune_folder_list(path_from_index):
    '''
    Given a list of *reified* paths to folders,  
    eliminate any paths contained within other paths.
    '''

    # Sort them to save work later.
    path_from_sorted_index = sorted(path_from_index, key=len)

    # For each path, see if any of the later/longer paths are a child path.
    # If a path is a child of another, mark the child for deletion.
    count = len(path_from_sorted_index)
    do_keep_from_sorted_index = [True] * count 
    for parent_sorted_index in range(count):
        parent_path = path_from_sorted_index[parent_sorted_index]
        for child_sorted_index in range(parent_sorted_index+1, count):
            if do_keep_from_sorted_index[child_sorted_index]:
                child_path = path_from_sorted_index[child_sorted_index]
                do_keep = ( os.path.commonpath([parent_path, child_path]) != os.path.commonpath([parent_path]) )
                  # The os.path.commonpath([parent_path]) is to protect from trailing slashes in parent_path.
                do_keep_from_sorted_index[child_sorted_index] = do_keep

    # Only keep the keepers    
    result = select(path_from_sorted_index, do_keep_from_sorted_index)
    return result
    


def mount_folder_list_from_apt_track_args(reified_apt_track_args):
    '''
    Given a dict of APT_Track.pt arguments, with all file paths already
    converted to 'reified' paths, outputs a list of folders that will need to be
    mounted. Note that the list may contain repeats and folders that are
    contained within other folders on the list.
    '''
    mount_folder_list_list = [ mount_folder_list_from_reified_apt_arg_dict_pair(key, value) for key,value in reified_apt_track_args.items() ]
    mount_folder_list = flatten(mount_folder_list_list)
    return mount_folder_list



# def mount_string_lists_from_apt_arg_dict_pair(name, value):
#     # Given a single (name, value) pair from the suppied arguments to APT_track, 
#     # Generate a list of mount strings that will be used to make sure docker can 
#     # access the needed files.
#     if name in file_name_arg_keys:
#         if isinstance(value, list):
#             result = [ mount_string_from_file_name(file_name) for file_name in value ]
#         elif value is None:
#             result = []
#         else:
#             file_name = value
#             result = [ mount_string_from_file_name(file_name) ]
#     else:
#         result = []
#     return result



def mount_tokens_from_folder_list(mount_folder_list, container_mount_root_path):
    '''
    From a list of absolute folder paths, make a list of tokens to use in the call to
    docker.
    '''
    raw_mount_strings = [ mount_string_from_folder_path(folder_path, container_mount_root_path) for folder_path in mount_folder_list ]
    mount_strings = flatten(raw_mount_strings)
    mount_tokens_as_list_of_lists = [ ['--mount', mount_string] for mount_string in mount_strings ]
    mount_tokens = flatten(mount_tokens_as_list_of_lists)
    return mount_tokens



# def mount_tokens_from_apt_track_args(apt_track_args):
#     # Given a dict of APT_Track.pt arguments, outputs a list of command-line tokens to be used when
#     # calling docker run, to ensure that all the needed files are available in the container.
#     # We call this a list of "mount tokens".
#     raw_mount_strings = [ mount_string_lists_from_apt_arg_dict_pair(name, value) for name,value in apt_track_args.items() ]
#     mount_strings = flatten(raw_mount_strings)
#     mount_tokens_as_list_of_lists = [ ['--mount', mount_string] for mount_string in mount_strings ]
#     mount_tokens = flatten(mount_tokens_as_list_of_lists)
#     return mount_tokens



def reify_paths_in_apt_arg_dict(apt_arg_dict):
    '''
    Convert all the file/dir names in apt_arg_dict to
    os.path.realpath()'ed versions.  Note this does not modify the input.
    '''
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



def containerize_paths_in_apt_arg_dict(reified_apt_arg_dict, container_mount_root_path):
    '''
    Convert all the file/dir names in apt_arg_dict to the corresponding path in
    the container.  We assume the input is reified.
    '''
    result = {}
    for key,value in reified_apt_arg_dict.items():        
        if key in file_name_arg_keys:
            if isinstance(value, list):
                new_value = [ containerize_path(path, container_mount_root_path) for path in value ]
            elif value is None:
                new_value = None
            else:
                new_value = containerize_path(value, container_mount_root_path)        
        else:
            new_value = value
        result[key] = new_value
    return result



def apt_track_tokens_from_apt_track_args_helper(key, value):
    '''
    Helper function for creating a list of tokens from the args dict returned by
    parse_args().  Converts a single key-value pair to the corresponding token
    list.
    '''
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

    # print('')
    # print('key:')
    # print(key)
    # print('value:')
    # print(value)
    # print('result:')
    # print(result)

    return result



def apt_track_tokens_from_apt_track_args(apt_track_args):
    '''
    Create a list of tokens from the args dict returned by parse_args().
    '''
    protoresult = [ apt_track_tokens_from_apt_track_args_helper(key, value) for key, value in apt_track_args.items() ]
    result = flatten(protoresult)
    return result



def run_with_conda(environment_name, apt_track_py_absolute_path, reified_apt_track_args):
    #environment_name = 'apt_20230427_tf211_pytorch113_ampere'
    # Remake the tokens to be used in the call to APT_track.py, to use the reified paths
    apt_track_tokens = apt_track_tokens_from_apt_track_args(reified_apt_track_args)
    command_as_list = ['conda', 'run', '--live-stream', '-n', environment_name, 'python', apt_track_py_absolute_path] + apt_track_tokens
    subprocess.run(command_as_list, check=True)



def run_with_docker(image_tag, apt_track_py_absolute_path, reified_apt_track_args):
    # Specify the tag of the image to use
    #image_tag = 'bransonlabapt/apt_docker:apt_20230427_tf211_pytorch113_ampere'

    # # Get the '--mount' command-line tokens from the APT_track arguments 
    # apt_track_mount_tokens = mount_tokens_from_apt_track_args(reified_apt_track_args)

    # Extract a list of the folders to mount, with no repeats or contained folders
    arguments_mount_folder_list = mount_folder_list_from_apt_track_args(reified_apt_track_args)

    # Get the '--mount' command-line tokens for the APT deepnet/ folder
    deepnet_folder_absolute_path = os.path.normpath(os.path.dirname(apt_track_py_absolute_path))

    # Get the working folder path
    container_mount_root_path = '/mnt'
    working_folder_path = os.path.normpath(os.path.realpath(os.getcwd()))
    containerized_working_folder_path = containerize_path(working_folder_path, container_mount_root_path)

    # Get the home folder path, and the in-container version
    home_folder_path = os.path.normpath(os.path.realpath(os.getenv('HOME')))
    containerized_home_folder_path = containerize_path(home_folder_path, container_mount_root_path)

    # Finalize the list of mounts
    raw_mount_folder_list = arguments_mount_folder_list + [ deepnet_folder_absolute_path, working_folder_path, home_folder_path ]
    mount_folder_list = prune_folder_list(raw_mount_folder_list)

    # Make the list of tokens to use in the docker call
    mount_tokens = mount_tokens_from_folder_list(mount_folder_list, container_mount_root_path)

    # Convert paths in the args to the in-container versions
    containerized_apt_track_args = containerize_paths_in_apt_arg_dict(reified_apt_track_args, container_mount_root_path)
    # print('containerized_apt_track_args:')
    # print(containerized_apt_track_args)

    # Remake the tokens to be used in the call to APT_track.py, to use the reified paths
    apt_track_tokens = apt_track_tokens_from_apt_track_args(containerized_apt_track_args)

    # Translate the APT_track.py path to the container-side version
    containerized_apt_track_py_absolute_path = containerize_path(apt_track_py_absolute_path, container_mount_root_path)

    # Get the uid+gid, and create the --user arg string
    uid = os.getuid()
    gid = os.getgid()
    user_argument_string = '%d:%d' % (uid,gid)        

    # Assemble the command line (as a list of tokens)
    command_as_list = ['docker', 'run'] +\
                      ['--rm', '--ipc=host', '--network', 'host', '--shm-size=8G' ] + \
                      ['--user', user_argument_string, '--gpus', 'all' ] + \
                      ['--workdir', containerized_working_folder_path] + \
                      ['-e', 'USER='+os.getenv('USER')] + \
                      ['-e', 'HOME='+containerized_home_folder_path] + \
                      mount_tokens + \
                      [ image_tag, 'python', containerized_apt_track_py_absolute_path ] + \
                      apt_track_tokens
    # print('command_as_list:')
    # print(command_as_list)                  

    # Finally, execute the command line
    subprocess.run(command_as_list, check=True)



def run_with_apptainer(image_spec, apt_track_py_absolute_path, reified_apt_track_args):
    # Specify the tag of the image to use
    #image_spec = 'docker://bransonlabapt/apt_docker:apt_20230427_tf211_pytorch113_ampere'

    # # Get the '--mount' command-line tokens from the APT_track arguments 
    # apt_track_mount_tokens = mount_tokens_from_apt_track_args(reified_apt_track_args)

    # Extract a list of the folders to mount, with no repeats or contained folders
    arguments_mount_folder_list = mount_folder_list_from_apt_track_args(reified_apt_track_args)

    # Get the '--mount' command-line tokens for the APT deepnet/ folder
    deepnet_folder_absolute_path = os.path.normpath(os.path.dirname(apt_track_py_absolute_path))

    # Get the working folder path
    container_mount_root_path = '/mnt'
    working_folder_path = os.path.normpath(os.path.realpath(os.getcwd()))
    containerized_working_folder_path = containerize_path(working_folder_path, container_mount_root_path)

    # Get the home folder path, and the in-container version
    home_folder_path = os.path.normpath(os.path.realpath(os.getenv('HOME')))
    # containerized_home_folder_path = containerize_path(home_folder_path, container_mount_root_path)

    # Finalize the list of mounts
    raw_mount_folder_list = arguments_mount_folder_list + [ deepnet_folder_absolute_path, working_folder_path, home_folder_path ]
    mount_folder_list = prune_folder_list(raw_mount_folder_list)

    # Make the list of tokens to use in the docker call
    mount_tokens = mount_tokens_from_folder_list(mount_folder_list, container_mount_root_path)

    # Convert paths in the args to the in-container versions
    containerized_apt_track_args = containerize_paths_in_apt_arg_dict(reified_apt_track_args, container_mount_root_path)
    # print('containerized_apt_track_args:')
    # print(containerized_apt_track_args)

    # Remake the tokens to be used in the call to APT_track.py, to use the reified paths
    apt_track_tokens = apt_track_tokens_from_apt_track_args(containerized_apt_track_args)

    # Translate the APT_track.py path to the container-side version
    containerized_apt_track_py_absolute_path = containerize_path(apt_track_py_absolute_path, container_mount_root_path)

    # # Get the uid+gid, and create the --user arg string
    # uid = os.getuid()
    # gid = os.getgid()
    # user_argument_string = '%d:%d' % (uid,gid)        

    # Assemble the command line (as a list of tokens)
    command_as_list = ['apptainer', 'run'] + \
                      ['--nv' ] + \
                      ['--workdir', containerized_working_folder_path] + \
                      mount_tokens + \
                      [ image_spec, 'python', containerized_apt_track_py_absolute_path ] + \
                      apt_track_tokens
    # print('command_as_list:')
    # print(command_as_list)                  

    # Finally, execute the command line
    subprocess.run(command_as_list, check=True)



def main(argv):
    # Get an absolute path to APT_track.py
    this_script_path = os.path.realpath(__file__)
    this_script_folder_path = os.path.dirname(this_script_path)
    apt_track_py_absolute_path = os.path.join(this_script_folder_path, 'APT_track.py')

    (args, apt_track_args) = parse_args(argv)
       # args is a namespace, leftover_tokens is a list of strings
    # print('args:')
    # print(args) 
    # print('apt_track_args:')
    # print(apt_track_args)

    # Convert the paths in apt_track_args to realpath'ed paths
    reified_apt_track_args = reify_paths_in_apt_arg_dict(apt_track_args)
    # print('reified_apt_track_args:')
    # print(reified_apt_track_args)

    do_use_conda_backend = not (args['conda'] is None)
    do_use_docker_backend = not (args['docker'] is None)
    do_use_apptainer_backend = not (args['apptainer'] is None)
    backend_count = int(do_use_conda_backend) + int(do_use_docker_backend) + int(do_use_apptainer_backend)
    if backend_count==0:
        backend = 'conda'
        backend_spec = conda_default_environment_name
    elif backend_count==1:
        if do_use_conda_backend:
            backend = 'conda'
            backend_spec = args['conda']
        elif do_use_docker_backend:
            backend = 'docker'
            backend_spec = args['docker']
        elif do_use_apptainer_backend:
            backend = 'apptainer'
            backend_spec = args['apptainer']
        else:
            raise RuntimeError('Internal error: Exactly one backend specified, but it is seemingly not conda, docker, or apptainer.')
    else:
        raise RuntimeError('More than one backend specified.  You can''t do that.')

    if backend=='conda':
        run_with_conda(backend_spec, apt_track_py_absolute_path, reified_apt_track_args)
    elif backend=='docker':
        run_with_docker(backend_spec, apt_track_py_absolute_path, reified_apt_track_args)
    elif backend=='apptainer':
        run_with_apptainer(backend_spec, apt_track_py_absolute_path, reified_apt_track_args)
    else:
        raise RuntimeError('Internal error: The backend does not seem to be either conda, docker, or apptainer.')



if __name__ == "__main__":
    main(sys.argv[1:])
