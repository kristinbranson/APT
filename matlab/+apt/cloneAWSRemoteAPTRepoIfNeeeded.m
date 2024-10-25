function cloneAWSRemoteAPTRepoIfNeeeded(remote_apt_root)
  % Clone 'remote' repo into cacheRoot from prod, if necessary
  % 
  % cacheRoot: 'remote' cachedir, ie cachedir on JRC filesys
  
  % does repo in 'remote' cache exist?
  command_line_1 = sprintf('bash -c "[ -d ''%s'' ] && echo ''y'' || echo ''n''"',remote_apt_root);
  
  command_line_1 = wrapCommandSSH(command_line_1,'host',DLBackEndClass.jrchost);
  
  [~,res] = apt.syscmd(command_line_1,...
                       'failbehavior','err');
  res = strtrim(res);
  
  % clone it if nec
  switch res
    case 'y'
      fprintf('Found JRC/APT repo at %s.\n',remote_apt_root);
    case 'n'
      cloneaptcmd = sprintf('git clone %s %s',DLBackEndClass.jrcprodrepo,remote_apt_root);
      cloneaptcmd = wrapCommandSSH(cloneaptcmd,'host',DLBackEndClass.jrchost);
      apt.syscmd(cloneaptcmd,'failbehavior','err');
      fprintf('Cloned JRC/APT repo into %s.\n',remote_apt_root);
    otherwise
      error('Failed to update APT repo on JRC filesystem.');
  end
end
      
