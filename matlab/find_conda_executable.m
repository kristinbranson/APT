function result = find_conda_executable()
% Attempt to find the path to the conda executable.

persistent memoized_result

if isempty(memoized_result) ,
  apt_conda_path = '/opt/conda/condabin/conda' ;
  if exist(apt_conda_path, 'file') ,
    % On Ubuntu, if you install conda via apt (apt as in Ubuntu/Debian apt-get,
    % not APT as in Adavanced Part Tracker), we use that conda.
    memoized_result = apt_conda_path ;
  else
    % First check if conda *is* on the path
    [retval, location] = system('which conda') ;
    is_conda_on_path = (retval==0) && ~isempty(location) ;
    if is_conda_on_path ,
      memoized_result = strtrim(location) ;
    else
      % conda is not on the path

      % If CONDA_PREFIX is set, that will tell us where to find condabin/conda
      did_find_conda_prefix = false ;
      if isenv('CONDA_PREFIX')
        conda_prefix = getenv('CONDA_PREFIX') ;
        did_find_conda_prefix = true ;
      else
        home_path = getenv('HOME') ;
        folders_to_check_for = {'miniforge3' 'anaconda3', 'anaconda2', 'miniconda3', 'miniconda2', '.miniconda3-blurgh'} ;
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
          memoized_result = condabin_conda_path ;
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
end

result = memoized_result ;
