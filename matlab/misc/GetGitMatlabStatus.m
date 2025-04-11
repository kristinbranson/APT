function info = GetGitMatlabStatus(source_repo_folder_path)
    % Extract a bunch of git version info for the given source folder.
    % Also get the Matlab version.
    % This function reads the filesystem, but does not write to it.
    
    % Get the matlab version
    matlab_ver_string = version() ;
    
    % Figure out if the git command is available
    try
      system_with_error_handling('git --version');
      is_git_available = true;
    catch
      is_git_available = false;
    end

    % Determine the command preamble we'll use for many things
    preamble = sprintf('cd %s && ',source_repo_folder_path);
    if isunix()
      preamble = [preamble,'env GIT_SSL_NO_VERIFY=true GIT_TERMINAL_PROMPT=0 '];
    end

    % Get the git hash, etc    
    if ~is_git_available,
      source_repo_folder_path = 'unknown';
      commit_hash = 'unknown';
      git_remote_report = 'unknown';
      git_status = 'unknown';
    else
      % Get the commit hash
      commit_hash = strtrim(system_with_error_handling([preamble,'git rev-parse --verify HEAD']));
      
      % Get the git remote report
      git_remote_report = strtrim(system_with_error_handling([preamble,'git remote -v']));
      
      % Get the git status
      git_status = strtrim(system_with_error_handling([preamble,'git status -u no']));      
    end

    % Package everything up in a struct
    info = struct() ;
    info.matlab_ver_string = matlab_ver_string;
    info.source_repo_folder_path = source_repo_folder_path;
    info.commit_hash = commit_hash;
    info.git_remote_report = git_remote_report;
    info.git_status = git_status;    
end
