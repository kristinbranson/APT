function aptroot = bsubSetRootUpdateRepo(deepnetrunlocal)
  if deepnetrunlocal
    aptroot = APT.Root;
  else
    error('At present, for the JRC backend, the APT cache dir must be on the cluster file system') ;
    %apt.cloneJRCRepoIfNec(localCacheDir);
    %apt.updateAPTRepoExecJRC(localCacheDir);
    %aptroot = [localCacheDir '/APT'];
  end
  apt.cpupdatePTWfromJRCProdExec(aptroot);
end
