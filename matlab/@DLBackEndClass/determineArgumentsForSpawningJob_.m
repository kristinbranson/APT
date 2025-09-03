function result = determineArgumentsForSpawningJob_(obj, tracker, gpuid, jobinfo, aptroot, train_or_track)
  % Get arguments for a particular (spawning) job to be registered.
  
  if isequal(train_or_track,'train'),
    dmc = jobinfo;
    containerName = DeepModelChainOnDisk.getCheckSingle(dmc.trainContainerName);
  else
    dmc = jobinfo.trainDMC;
    containerName = jobinfo.containerName;
  end
  
  switch obj.type
    case DLBackEnd.Bsub
      mntPaths = obj.genContainerMountPathBsubDocker_(tracker,train_or_track,jobinfo);
      if isequal(train_or_track,'train'),
        native_log_file_path = DeepModelChainOnDisk.getCheckSingle(dmc.trainLogLnx);
        nslots = obj.jrcnslots;
      else % track
        native_log_file_path = jobinfo.logfile;
        nslots = obj.jrcnslotstrack;
      end
  
      % for printing git status? not sure why this is only in bsub and
      % not others. 
      aptrepo = DeepModelChainOnDisk.getCheckSingle(dmc.aptRepoSnapshotLnx());
      extraprefix = DeepTracker.repoSnapshotCmd(aptroot,aptrepo);
      singimg = obj.singularity_image_path;
  
      additionalBsubArgs = obj.jrcAdditionalBsubArgs ;
      result = {...
        'singargs',{'bindpath',mntPaths,'singimg',singimg},...
        'bsubargs',{'gpuqueue' obj.jrcgpuqueue 'nslots' nslots,'logfile',native_log_file_path,'jobname',containerName, ...
                    'additionalArgs', additionalBsubArgs},...
        'sshargs',{'extraprefix',extraprefix}...
        };
    case DLBackEnd.Docker
      mntPaths = obj.genContainerMountPathBsubDocker_(tracker,train_or_track,jobinfo);
      isgpu = ~isempty(gpuid) && ~isnan(gpuid);
      % tfRequiresTrnPack is always true;
      shmsize = 8;
      result = {...
        'containerName',containerName,...
        'bindpath',mntPaths,...
        'isgpu',isgpu,...
        'gpuid',gpuid,...
        'shmsize',shmsize...
        };
    case DLBackEnd.Conda
      if isequal(train_or_track,'train'),
        native_log_file_path = DeepModelChainOnDisk.getCheckSingle(dmc.trainLogLnx);
      else % track
        native_log_file_path = jobinfo.logfile;
      end
      result = {...
        'logfile', native_log_file_path };
      % backEndArgs = {...
      %   'condaEnv', obj.condaEnv, ...
      %   'gpuid', gpuid };
    case DLBackEnd.AWS
      result  = {} ;
    otherwise,
      error('Internal error: Unknown backend type') ;
  end
end  % function