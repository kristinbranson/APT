function cmd = wrapCommandToBeSpawnedForBackend(backend, basecmd, varargin)
  switch backend.type,
    case DLBackEnd.AWS
      cmd = wrapCommandToBeSpawnedForAWSBackend(backend, basecmd, varargin{:});
    case DLBackEnd.Bsub,
      cmd = wrapCommandToBeSpawnedForBsubBackend(backend, basecmd, varargin{:});
    case DLBackEnd.Conda
      cmd = wrapCommandToBeSpawnedForCondaBackend(backend, basecmd, varargin{:});
    case DLBackEnd.Docker
      cmd = wrapCommandToBeSpawnedForDockerBackend(backend, basecmd, varargin{:});
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
  bindpath = {'/home'} ;
  codestr = ...
    wrapCommandDocker(basecmd, ...
                      'dockerimg',dockerimg, ...
                      'bindpath',bindpath, ...
                      dockerargs{:}) ;

  % Wrap for ssh'ing into a remote docker host, if needed
  result = backend.wrapCommandSSHAWS(codestr, sshargs{:}) ;
end



function cmd = wrapCommandToBeSpawnedForBsubBackend(backend, basecmd, varargin)  %#ok<INUSD> 
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



function result = wrapCommandToBeSpawnedForCondaBackend(backend, basecmd, varargin)
  % Take a base command and run it in a conda env
  preresult = wrapCommandConda(basecmd, 'condaEnv', backend.condaEnv, 'logfile', '/dev/null', 'gpuid', backend.gpuids(1), varargin{:}) ;
  result = sprintf('{ %s & } && echo $!', preresult) ;  % echo $! to get the PID
end



function codestr = wrapCommandToBeSpawnedForDockerBackend(backend, basecmd, varargin)
  % Take a base command and run it in a docker img

  % Determine the fallback gpuid, keeping in mind that backend.gpuids may be
  % empty.
  if isempty(backend.gpuids) ,
    fallback_gpuid = 1 ;
  else
    fallback_gpuid = backend.gpuids(1) ;
  end    

  % Call main function, returns Linux/WSL-style command string
  codestr = ...
    wrapCommandDocker(basecmd, ...
                      'dockerimg',backend.dockerimgfull,...
                      'gpuid',fallback_gpuid,...
                      'apiver',apt.docker_api_version(), ...
                      varargin{:}) ;  % key-value pairs in varagin will override ones specified here

  % Wrap for ssh'ing into a remote docker host, if needed
  if ~isempty(backend.dockerremotehost),
    codestr = wrapCommandSSH(codestr,'host',backend.dockerremotehost);
  end
end  % function 
    
