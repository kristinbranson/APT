function result = wrapCommandConda(baseCommand, varargin)
% Take a base command and run it in a conda env.  baseCommand should be a
% ShellCommand in the wsl locale.

% Validate input baseCommand
assert(isa(baseCommand, 'apt.ShellCommand'), 'baseCommand must be an apt.ShellCommand object');
assert(baseCommand.tfDoesMatchLocale(apt.PathLocale.wsl), 'baseCommand must have WSL locale');

% Process keyword arguments
[condaEnv,gpuid,logfile] = ...
  myparse(varargin,...
          'condaEnv',[],...
          'gpuid',0, ...
          'logfile','');
if isempty(condaEnv) ,
  error('condaEnv argument is mandatory') ;
end

% Find the conda executable
condaExecutablePathNativeChar = find_conda_executable() ;
condaExecutablePathNative = apt.MetaPath(condaExecutablePathNativeChar, 'native', 'universal');
condaExecutablePathWsl = condaExecutablePathNative.asWsl();

% Augment the command with a specification of the GPU id, if called for
if isempty(gpuid) || isnan(gpuid) 
  partialCommand = baseCommand ;
else
  cudaDeviceOrderVar = apt.ShellVariableAssignment('CUDA_DEVICE_ORDER', 'PCI_BUS_ID');
  cudaVisibleDevicesVar = apt.ShellVariableAssignment('CUDA_VISIBLE_DEVICES', num2str(gpuid));
  cudaEnvCommand = apt.ShellCommand({'export', cudaDeviceOrderVar, '&&', 'export', cudaVisibleDevicesVar, '&&'}, apt.PathLocale.wsl, apt.Platform.posix);
  partialCommand = apt.ShellCommand.cat(cudaEnvCommand, baseCommand);
end

% Add logging
if isempty(logfile) 
  fullCommand = partialCommand ;
else
  logFilePathNative = apt.MetaPath(logfile, 'native', 'cache');
  logFilePathWsl = logFilePathNative.asWsl();
  fullCommand = partialCommand.append('&>', logFilePathWsl) ;
end

% The command will use the conda "run" subcommand, and use bash explicitly to
% interpret the command line using sequential ShellCommand objects
command0 = apt.ShellCommand({condaExecutablePathWsl, 'run', '-n', condaEnv, '/bin/bash', '-c'}, apt.PathLocale.wsl, apt.Platform.posix);
command1 = command0.append(fullCommand);

% Clear annoying Matlab envars
result = prependStuffToClearMatlabEnvironment(command1);
