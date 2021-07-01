function [breadcrumb_string,info] = GetGitMatlabStatus(source_repo_folder_path)

%     original_pwd = pwd() ;
%     cleaner = onCleanup(@()(cd(original_pwd))) ;
%     cd(source_repo_folder_path) ;
    
    % Get the matalb version
    matlab_ver_string = version() ;
    
    % This is hard to get working in a way that overrides
    % 'url."git@github.com:".insteadOf https://github.com/' for a single command.
    % Plus it hits github every time you run, which seems fragile...
    % % Make sure the git remote is up-to-date
    % system_with_error_handling('env GIT_SSL_NO_VERIFY=true GIT_TERMINAL_PROMPT=0 git remote update') ;    
    
    % Get the git hash
    
    preamble = sprintf('cd %s && ',source_repo_folder_path);
    if isunix
      preamble = [preamble,'env GIT_SSL_NO_VERIFY=true GIT_TERMINAL_PROMPT=0 '];
    end
    try
      system_with_error_handling('git --version');
      isgit = true;
    catch
      isgit = false;
    end
    
    if ~isgit,
      source_repo_folder_path = 'unknown';
      commit_hash = 'unknown';
      git_remote_report = 'unknown';
      git_status = 'unknown';
    else
      commit_hash = strtrim(system_with_error_handling([preamble,'git rev-parse --verify HEAD']));
      
      % Get the git remote report
      git_remote_report = strtrim(system_with_error_handling([preamble,'git remote -v']));
      
      % Get the git status
      git_status = strtrim(system_with_error_handling([preamble,'git status -u no']));
      
      % Get the recent git log
      %git_log = system_with_error_handling('env GIT_SSL_NO_VERIFY=true GIT_TERMINAL_PROMPT=0 git log --graph --oneline --max-count 10 | cat') ;
    end
        
    info.matlab_ver_string = matlab_ver_string;
    info.source_repo_folder_path = source_repo_folder_path;
    info.commit_hash = commit_hash;
    info.git_remote_report = git_remote_report;
    info.git_status = git_status;
    
    % Write a file with the commit hash into the folder, for good measure
    breadcrumb_string = GitMatlabBreadCrumbString(info);
                              
end
