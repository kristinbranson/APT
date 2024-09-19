function cmd = wrapCommandToBeSpawnedForBackend(backend, basecmd, varargin)
  switch backend.type,
    case DLBackEnd.AWS
      cmd = wrapCommandToBeSpawnedForAWSBackend(backend, basecmd, varargin{:});
    case DLBackEnd.Bsub,
      cmd = wrapCommandToBeSpawnedForBsubBackend(basecmd, varargin{:});
    case DLBackEnd.Conda
      cmd = wrapCommandToBeSpawnedForCondaBackend(basecmd, varargin{:});
    case DLBackEnd.Docker
      cmd = wrapCommandToBeSpawnedForDockerBackend(basecmd, varargin{:});
    otherwise
      error('Not implemented: %s',backend.type);
  end
end



function result = wrapCommandToBeSpawnedForAWSBackend(backend, basecmd, varargin)
  % Wrap for docker, returns Linux/WSL-style command string
  
  % Parse arguments
  [dockerargs, sshargs] = ...
    myparse_nocheck(varargin, ...
                    'dockerargs',{}, ...
                    'sshargs',{}) ;

  % Wrap for docker
  dockerimg = 'bransonlabapt/apt_docker:apt_20230427_tf211_pytorch113_ampere' ;
  bindpath = {} ;
  codestr = ...
    wrapCommandDocker(basecmd, ...
                      'dockerimg',dockerimg, ...
                      'bindpath',bindpath, ...
                      dockerargs{:}) ;

  % Wrap for ssh'ing into a remote docker host, if needed
  result = backend.awsec2.wrapCommandSSH(codestr, sshargs{:}) ;
end



function cmd = wrapCommandToBeSpawnedForBsubBackend(basecmd, varargin)
  [singargs,bsubargs,sshargs] = myparse(varargin,'singargs',{},'bsubargs',{},'sshargs',{});
  cmd1 = wrapCommandSing(basecmd,singargs{:});
  cmd2 = wrapCommandBsub(cmd1,bsubargs{:});

  % already on cluster?
  tfOnCluster = ~isempty(getenv('LSB_DJOB_NUMPROC'));
  if tfOnCluster,
    % The Matlab environment vars cause problems with e.g. PyTorch
    cmd = prepend_stuff_to_clear_matlab_environment(cmd2) ;
  else
    % Doing ssh does not pass Matlab envars, so they don't cause problems in this case.  
    cmd = wrapCommandSSH(cmd2,'host',DLBackEndClass.jrchost,sshargs{:});
  end
end



function result = wrapCommandToBeSpawnedForCondaBackend(basecmd, varargin)
  % Take a base command and run it in a sing img
  [condaEnv,logfile,gpuid] = ...
    myparse(varargin,...
            'condaEnv',DLBackEndClass.default_conda_env, ...
            'logfile', '/dev/null', ...
            'gpuid',0);
  preresult = wrapCommandConda(basecmd, 'condaEnv', condaEnv, 'logfile', logfile, 'gpuid', gpuid) ;
  result = sprintf('{ %s & } && echo $!', preresult) ;  % echo $! to get the PID
end



function codestr = wrapCommandToBeSpawnedForDockerBackend(basecmd, varargin)
  % Take a base command and run it in a docker img

  % Parse keyword args
  [containerName,bindpath,isgpu,gpuid,tfDetach,tty,shmSize] = ...
    myparse(varargin,...
            'containername','aptainer',...
            'bindpath',{},... % paths on local filesystem that must be mounted/bound within container
            'isgpu',true,... % set to false for CPU-only
            'gpuid',0,... % used if isgpu
            'detach',true, ...
            'tty',false,...
            'shmsize',[] ... optional
            );

  % Call main function, returns Linux/WSL-style command string
  codestr = ...
    wrapCommandDocker(basecmd, ...
                      'containername',containerName,...
                      'bindpath',bindpath,...
                      'dockerimg',obj.dockerimgfull,...
                      'isgpu',isgpu,...
                      'gpuid',gpuid,...
                      'detach',tfDetach, ...
                      'tty',tty,...
                      'shmsize',shmSize, ...
                      'apiver',obj.dockerapiver) ;
end  % function 
    
