function command_string = synthesize_conda_command(conda_args_as_string)
% Synthesize a conda command string, trying to be smart about Linux/Windows, and exactly how conda
% was installed.  The intent is that this command string will be passed to
% system().

apt_conda_path = '/opt/conda/condabin/conda' ;
apt_conda_init_script_path = '/opt/conda/etc/profile.d/conda.sh' ;
if exist(apt_conda_path, 'file') && exist(apt_conda_init_script_path, 'file') ,
  % On Ubuntu, if you install conda via apt (apt as in Ubuntu/Debian apt-get,
  % not APT as in Adavanced Part Tracker), we source the files that set up the
  % shell variables so that e.g. conda activate works properly.
  % .bashrc is not sourced when we call system() b/c the shell is
  % noninteractive.
  %command_string = sprintf('source %s && conda %s', apt_conda_init_script_path, conda_args_as_string) ;
  enable_conda_command = sprintf('source %s', apt_conda_init_script_path) ;
else
  % Because system() commands run in a noninteractive shell, .bashrc is not
  % sourced, so conda is typically not even on the path with the standard miniconda
  % install.

  % First check if conda *is* on the path
  [retval, location] = system('which conda') ;
  is_conda_on_path = (retval==0) && ~isempty(location) ;
  if is_conda_on_path ,
    enable_conda_command = 'eval "$(conda shell.bash hook)"' ;
  else
    % conda is not on the path

    % If CONDA_PREFIX is set, that will tell us where to find condabin/conda,
    % which we can use to enable conda in the shell
    did_find_conda_prefix = false ;
    if isenv('CONDA_PREFIX')
      conda_prefix = getenv('CONDA_PREFIX') ;
      did_find_conda_prefix = true ;
    else
      home_path = getenv('HOME') ;
      folders_to_check_for = {'anaconda3', 'anaconda2', 'miniconda3', 'miniconda2', '.miniconda3-blurgh'} ;
      for i = length(folders_to_check_for) ,
        folder_name = folders_to_check_for{i} ;
        conda_prefix = fullfile(home_path, folder_name) ;
        if exist(conda_prefix, 'dir') ,
          did_find_conda_prefix = true ;
          break
        end
      end
    end

    % Conda is the worst
    if did_find_conda_prefix ,
      % If we found the conda dir, use condabin/conda to setup conda
      condabin_conda_path = fullfile(conda_prefix, 'condabin', 'conda') ;
      if exist(condabin_conda_path, 'file') ,
        enable_conda_command = sprintf('eval "$(%s shell.bash hook)"', condabin_conda_path) ;
      else
        error('We thought we found conda installed in %s, but condabin/conda is missing', conda_prefix) ;
      end
    else
      % Error
      % We used to try sourcing .bashrc, but usually .bashrc has a line in it to
      % exit early if the shell is noninteractive, so we don't bother trying that
      % anymore.
      error('Unable to find condabin/conda') ;
    end
  end
end

% Finally, synthesize the final bash command line
command_string = sprintf('%s && conda %s', enable_conda_command, conda_args_as_string) ;

end
