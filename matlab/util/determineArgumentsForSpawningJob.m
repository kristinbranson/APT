function result = determineArgumentsForSpawningJob(backend,deeptracker,gpuid,jobinfo,aptroot,train_or_track)
% Get arguments for a particular (spawning) job to be registered.

if isequal(train_or_track,'train'),
  dmc = jobinfo;
  containerName = DeepModelChainOnDisk.getCheckSingle(dmc.trainContainerName);
else
  dmc = jobinfo.trainDMC;
  containerName = jobinfo.containerName;
end

switch backend.type
  case DLBackEnd.Bsub
    mntPaths = deeptracker.genContainerMountPathBsubDocker(backend,train_or_track,jobinfo);
    if isequal(train_or_track,'train'),
      logfile = DeepModelChainOnDisk.getCheckSingle(dmc.trainLogLnx);
      nslots = backend.jrcnslots;
    else % track
      logfile = jobinfo.logfile;
      nslots = backend.jrcnslotstrack;
    end

    % for printing git status? not sure why this is only in bsub and
    % not others. 
    aptrepo = DeepModelChainOnDisk.getCheckSingle(dmc.aptRepoSnapshotLnx());
    extraprefix = DeepTracker.repoSnapshotCmd(aptroot,aptrepo);
    singimg = deeptracker.singularityImgPath();

    additionalBsubArgs = backend.jrcAdditionalBsubArgs ;
    result = {...
      'singargs',{'bindpath',mntPaths,'singimg',singimg},...
      'bsubargs',{'gpuqueue' backend.jrcgpuqueue 'nslots' nslots,'logfile',logfile,'jobname',containerName, ...
                  'additionalArgs', additionalBsubArgs},...
      'sshargs',{'extraprefix',extraprefix}...
      };
  case DLBackEnd.Docker
    mntPaths = deeptracker.genContainerMountPathBsubDocker(backend,train_or_track,jobinfo);
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
      logfile = DeepModelChainOnDisk.getCheckSingle(dmc.trainLogLnx);
    else % track
      logfile = jobinfo.logfile;
    end
    result = {...
      'logfile', logfile };
    % backEndArgs = {...
    %   'condaEnv', obj.condaEnv, ...
    %   'gpuid', gpuid };
  case DLBackEnd.AWS
    result  = {} ;
  otherwise,
    error('Internal error: Unknown backend type') ;
end
