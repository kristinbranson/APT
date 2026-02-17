function result = wrapCommandDocker(baseCommand, varargin)
  % Take a linux/WSL shell command, wrap it for running in a docker
  % container.  baseCommand should be a ShellCommand with WSL locale.
  % This function does not handle the case of running docker remotely via ssh.

  % Validate input baseCommand
  assert(isa(baseCommand, 'apt.ShellCommand'), 'baseCommand must be an apt.ShellCommand object');
  assert(baseCommand.tfDoesMatchLocale(apt.PathLocale.wsl), 'baseCommand must have WSL locale');

  % Parse keyword args
  [containerName,bindpath,dockerimg,isgpu,gpuid,tfDetach,tty,shmsize] = ...
    myparse(varargin,...
            'containername','apt-docker-container',...
            'bindpath',{},... % cell array of MetaPath objects, containing paths on local filesystem that must be mounted/bound within container
            'dockerimg','',... 
            'isgpu',true,... % set to false for CPU-only
            'gpuid',0,... % used if isgpu
            'detach',true, ...
            'tty',false,...
            'shmsize',[]) ;

  % Check mandatory keyword args
  if isempty(dockerimg) ,
    error('dockerimg cannot be empty') ;
  end
  
  % Get path to the deepnet/ subdirectory in the APT source tree
  aptDeepnetPathNativeChar = APT.getpathdl() ;  % this is a native, local path, as char
  aptDeepnetPathNative = apt.MetaPath(aptDeepnetPathNativeChar, 'native', 'source');
  aptDeepnetPathWsl = aptDeepnetPathNative.asWsl();

  % Set the paths to make visible in the container
  % Add whatever the user passed in as paths to bind to the container
  % Convert bindpath to MetaPath objects and create ShellBind objects
  if ~isempty(bindpath)
    bindPathWsl = cellfun(@(path) path.asWsl(), bindpath, 'UniformOutput', false);
    bindings = row(cellfun(@(src) apt.ShellBind(src, src), bindPathWsl, 'UniformOutput', false));
  else
    bindings = cell(1,0);
  end
  % Need the --mount in front of each binding
  mountArgsAsNestedList = cellfun(@(binding)({'--mount', binding}), bindings, 'UniformOutput', false);
  mountArgsAsList = flatten_row_cell_array(mountArgsAsNestedList) ;
  mountArgs = apt.ShellCommand(mountArgsAsList, apt.PathLocale.wsl, apt.Platform.posix);

  % Apparently we need to use the --user switch when running in real Linux
  % Actually think we want to use it regardless.  Otherwise on AWS files written
  % by things running in docker are owned by root, not by user "ubuntu".  This
  % makes it awkward to delete/overwrite them later.
  userArgs = apt.ShellCommand({'--user', '$(id -u):$(id -g)'}, apt.PathLocale.wsl, apt.Platform.posix);
  % if ispc() 
  %   userArgs = apt.ShellCommand({}, apt.PathLocale.wsl, apt.Platform.posix);
  % else
  %   userArgs = apt.ShellCommand({'--user', '$(id -u):$(id -g)'}, apt.PathLocale.wsl, apt.Platform.posix);
  % end    

  if isgpu
    %nvidiaArgs = {'--runtime nvidia'};
    gpuArgs = apt.ShellCommand({'--gpus', 'all'}, apt.PathLocale.wsl, apt.Platform.posix);
    cudaDeviceOrderVar = apt.ShellVariableAssignment('CUDA_DEVICE_ORDER', 'PCI_BUS_ID');
    cudaVisibleDevicesVar = apt.ShellVariableAssignment('CUDA_VISIBLE_DEVICES', num2str(gpuid));
    cudaEnv = apt.ShellCommand({'export', cudaDeviceOrderVar, ';', 'export', cudaVisibleDevicesVar, ';'}, apt.PathLocale.wsl, apt.Platform.posix);
  else
    gpuArgs = apt.ShellCommand({}, apt.PathLocale.wsl, apt.Platform.posix);
    cudaVisibleDevicesVar = apt.ShellVariableAssignment('CUDA_VISIBLE_DEVICES', '');
    cudaEnv = apt.ShellCommand({'export', cudaVisibleDevicesVar, ';'}, apt.PathLocale.wsl, apt.Platform.posix);
    % MK 20220411 We need to explicitly set devices for pytorch when not using GPUS
  end
  
  nativeHomeDirPathAsChar = get_home_dir_name() ;      
  nativeHomeDirPath = apt.MetaPath(nativeHomeDirPathAsChar, 'native', 'universal');
  homeDirWslPath = nativeHomeDirPath.asWsl();
  user = get_user_name() ;
  
  dockercmd = apt.dockercmd() ;

  if tfDetach,
    detachstr = '-d';
  else
    if tty
      detachstr = '-it';
    else
      detachstr = '-i';
    end        
  end
  
  if ~isempty(shmsize)
    otherArgs = apt.ShellCommand({sprintf('--shm-size=%dG',shmsize)}, apt.PathLocale.wsl, apt.Platform.posix);
  else
    otherArgs = apt.ShellCommand({}, apt.PathLocale.wsl, apt.Platform.posix);
  end

  command0 = ...
    apt.ShellCommand({dockercmd, 'run', detachstr, sprintf('--name %s',containerName), '--rm', '--ipc=host', '--network host'}, ...
                     apt.PathLocale.wsl, ...
                     apt.Platform.posix);
  command1 = command0.cat(mountArgs);
  command2 = command1.cat(gpuArgs);
  command3 = command2.cat(userArgs);
  command4 = command3.cat(otherArgs);
  userVar = apt.ShellVariableAssignment('USER', user);
  command5 = command4.append('-w', aptDeepnetPathWsl, '-e', userVar, dockerimg);
  
  homeVar = apt.ShellVariableAssignment('HOME', homeDirWslPath);
  command6 = apt.ShellCommand({'export', homeVar, ';'}, apt.PathLocale.wsl, apt.Platform.posix);
  subcommand = command6.cat(cudaEnv, 'cd', aptDeepnetPathWsl, ';', baseCommand);
  result = command5.append('bash', '-c', subcommand);
end  % function



% function result = mount_option_string(src, dst)
%   quoted_src = escape_string_for_bash(src) ;
%   quoted_dst = escape_string_for_bash(dst) ;
%   result = sprintf('--mount type=bind,src=%s,dst=%s',quoted_src,quoted_dst) ;
% end
