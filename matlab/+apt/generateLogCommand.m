function logcmd = generateLogCommand(backend, train_or_track, dmcjob_or_totrackinfojob)

if strcmp(train_or_track, 'train') ,
  dmcjob = dmcjob_or_totrackinfojob ;
  if backend.type == DLBackEnd.Docker ,
    containerName = DeepModelChainOnDisk.getCheckSingle(dmcjob.trainContainerName) ;
    logfile = DeepModelChainOnDisk.getCheckSingle(dmcjob.trainLogLnx) ;
    logcmd = generateLogCommandForDockerBackend(backend, containerName, logfile) ;
  else
    logcmd = '' ;
  end
elseif strcmp(train_or_track, 'track') ,
  totrackinfojob = dmcjob_or_totrackinfojob ;
  if backend.type == DLBackEnd.Docker ,
    containerName = totrackinfojob.containerName ;
    logfile = totrackinfojob.logfile ;
    logcmd = generateLogCommandForDockerBackend(backend, containerName, logfile) ;
  else
    logcmd = '' ;
  end
end
