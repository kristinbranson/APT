function updateAPTRepoExecJRC(cacheRoot) % throws if fails
  % cacheRoot: 'remote' cachedir, ie cachedir on JRC filesys
  updatecmd = apt.updateAPTRepoCmd('aptparent',cacheRoot);
  updatecmd = wrapCommandSSH(updatecmd,'host',DLBackEndClass.jrchost);
  [~,res] = DeepTracker.syscmd(updatecmd,...
    'failbehavior','err');
end

