function codestr = wrapCommandDocker(basecmd, varargin)
  % Take a linux/WSL shell command string, wrap it for running in a docker
  % container.  Returned string is also a linux/WSL shell command.  Any paths
  % handed to this function should be WSL paths.  This function does not handle
  % the case of running docker remotely via ssh.

  % Parse keyword args
  [containerName,bindpath,dockerimg,isgpu,gpuid,tfDetach,tty,shmsize,apiver] = ...
    myparse(varargin,...
            'containername','apt-docker-container',...
            'bindpath',{},... % paths on local filesystem that must be mounted/bound within container
            'dockerimg','',... 
            'isgpu',true,... % set to false for CPU-only
            'gpuid',0,... % used if isgpu
            'detach',true, ...
            'tty',false,...
            'shmsize',[], ... optional
            'apiver','1.40') ;

  % Check mandatory keyword args
  if isempty(dockerimg) ,
    error('dockerimg cannot be empty') ;
  end
  
  % Get path to the deepnet/ subdirectory in the APT source tree
  aptdeepnet = APT.getpathdl() ;  % this is a native, local path
  deepnetrootContainer = wsl_path_from_native(aptdeepnet) ;

  % Set the paths to make visible in the container
  % Add whatever the user passed in as paths to bind to the container
  mountArgs = cellfun(@mount_option_string,bindpath,bindpath,'uni',0);
  % if ispc() 
  %   % For running in WSL, want to add /mnt to the container, since user's files
  %   % are presumably in there.
  %   srcbindpath = {'/mnt'};
  %   dstbindpath = {'/mnt'};
  %   mountArgs = cellfun(@mount_option_string,srcbindpath,dstbindpath,'uni',0);
  % else    
  %   % Add whatever the user passed in as paths to bind to the container
  %   mountArgs = cellfun(@mount_option_string,bindpath,bindpath,'uni',0);
  % end

  % Apparently we need to use the --user switch when running in real Linux
  % Actually think we want to use it regardless.  Otherwise on AWS files written
  % by things running in docker are owned by root, not by user "ubuntu".  This
  % makes it awkward to delete/overwrite them later.
  userArgs = {'--user' '$(id -u):$(id -g)'};
  % if ispc() 
  %   userArgs = {};
  % else
  %   userArgs = {'--user' '$(id -u):$(id -g)'};
  % end    

  if isgpu
    %nvidiaArgs = {'--runtime nvidia'};
    gpuArgs = {'--gpus' 'all'};
    cudaEnv = sprintf('export CUDA_DEVICE_ORDER=PCI_BUS_ID; export CUDA_VISIBLE_DEVICES=%d;',gpuid);
  else
    gpuArgs = cell(1,0);
    cudaEnv = 'export CUDA_VISIBLE_DEVICES=;'; 
    % MK 20220411 We need to explicitly set devices for pytorch when not using GPUS
  end
  
  native_home_dir = get_home_dir_name() ;      
  user = get_user_name() ;
  
  dockercmd = dockercmd_from_apiver(apiver) ;

  if tfDetach,
    detachstr = '-d';
  else
    if tty
      detachstr = '-it';
    else
      detachstr = '-i';
    end        
  end
  
  otherargs = cell(0,1);
  if ~isempty(shmsize)
    otherargs{end+1,1} = sprintf('--shm-size=%dG',shmsize);
  end

  code_as_list = [
    {
    dockercmd
    'run'
    detachstr
    sprintf('--name %s',containerName);
    '--rm'
    '--ipc=host'
    '--network host'
    };
    mountArgs(:);
    gpuArgs(:);
    userArgs(:);
    otherargs(:);
    {
    '-w'
    escape_string_for_bash(deepnetrootContainer)
    '-e'
    ['USER=' user]
    dockerimg
    }
    ];
  linux_home_dir = wsl_path_from_native(native_home_dir) ;
  bashcmd = ...
    sprintf('export HOME=%s ; %s cd %s ; %s',...
            escape_string_for_bash(linux_home_dir), ...
            cudaEnv, ...
            escape_string_for_bash(deepnetrootContainer), ...
            basecmd) ;
  escbashcmd = ['bash -c ' escape_string_for_bash(bashcmd)] ;
  code_as_list{end+1} = escbashcmd;      
  codestr = sprintf('%s ',code_as_list{:});
  codestr = codestr(1:end-1);
end  % function



function result = mount_option_string(src, dst)
  quoted_src = escape_string_for_bash(src) ;
  quoted_dst = escape_string_for_bash(dst) ;
  result = sprintf('--mount type=bind,src=%s,dst=%s',quoted_src,quoted_dst) ;
end
