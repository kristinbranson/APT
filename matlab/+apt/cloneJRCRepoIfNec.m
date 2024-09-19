function cloneJRCRepoIfNec(cacheRoot)  % throws on fail
  % Clone 'remote' repo into cacheRoot from prod, if necessary
  % 
  % cacheRoot: 'remote' cachedir, ie cachedir on JRC filesys
  
  % does repo in 'remote' cache exist?
  aptroot = [cacheRoot '/APT'];
  aptrootexistscmd = sprintf('bash -c "[ -d ''%s'' ] && echo ''y'' || echo ''n''"',aptroot);
  aptrootexistscmd = wrapCommandSSH(aptrootexistscmd,'host',DLBackEndClass.jrchost);
  
  [~,res] = apt.syscmd(aptrootexistscmd,'failbehavior','err');
  res = strtrim(res);
  
  % clone it if nec
  switch res
    case 'y'
      fprintf('Found JRC/APT repo at %s.\n',aptroot);
    case 'n'
      cloneaptcmd = sprintf('git clone %s %s',DLBackEndClass.jrcprodrepo,aptroot);
      cloneaptcmd = wrapCommandSSH(cloneaptcmd,'host',DLBackEndClass.jrchost);
      apt.syscmd(cloneaptcmd,...
                 'failbehavior','err');
      fprintf('Cloned JRC/APT repo into %s.\n',aptroot);
    otherwise
      error('Failed to update APT repo on JRC filesystem.');
  end
end
      
