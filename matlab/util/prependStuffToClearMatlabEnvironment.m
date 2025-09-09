function result = prependStuffToClearMatlabEnvironment(inputCommand) 
% When you call system(), the envars are polluted with a bunch of 
% Matlab-specific things that can break stuff.  E.g. Matlab changes
% LD_LIBRARY_PATH, and that often breaks code you'd like to run with system().
% This prepends a bunch of unset and export commands to your command to
% fix these issues.
%
% Args:
%   inputCommand (apt.ShellCommand): Input command to prepend environment clearing to
%
% Returns:
%   apt.ShellCommand: Command with environment clearing prepended

% Validate input
assert(isa(inputCommand, 'apt.ShellCommand'), 'inputCommand must be an apt.ShellCommand object');

% Matlab sets all these envars, at least one of which seem to cause the PyTorch
% dataloader to segfault.  So we unset them all.
envarNamesToClear = ...
  { 'ARCH', 'AUTOMOUNT_MAP', 'BASEMATLABPATH', 'ICU_TIMEZONE_FILES_DIR', 'KMP_BLOCKTIME', 'KMP_HANDLE_SIGNALS', 'KMP_INIT_AT_FORK', ...
    'KMP_STACKSIZE', 'LC_NUMERIC', 'LD_PRELOAD', 'LIBVA_MESSAGING_LEVEL', 'MEMKIND_HEAP_MANAGER', 'MKL_DOMAIN_NUM_THREADS', ...
    'MKL_NUM_THREADS', 'OSG_LD_LIBRARY_PATH', 'PRE_LD_PRELOAD', 'TOOLBOX', 'XFILESEARCHPATH' } ;

% Create unset commands as a ShellCommand
unsetCommand = apt.ShellCommand({}, inputCommand.locale_, inputCommand.platform_);
for i = 1:numel(envarNamesToClear)
  if i == 1
    unsetCommand = unsetCommand.append('unset', envarNamesToClear{i});
  else
    unsetCommand = unsetCommand.append('&&', 'unset', envarNamesToClear{i});
  end
end

% We want to parse the LD_LIBRARY_PATH and purge it of any Matlab-related
% stuff.  If it's not set we can skip this.
if isenv('LD_LIBRARY_PATH') 
  orginalLdLibraryPath = getenv('LD_LIBRARY_PATH') ;
  dirFromOriginalPathIndex = strsplit(orginalLdLibraryPath, ':') ;
  isMatlabbyFromOriginalPathIndex = ...
      contains(dirFromOriginalPathIndex, 'matlab', 'IgnoreCase', true) | contains(dirFromOriginalPathIndex, 'mathworks', 'IgnoreCase', true) ;
  dirFromPathIndex = dirFromOriginalPathIndex(~isMatlabbyFromOriginalPathIndex) ;
  ldLibraryPath = strjoin(dirFromPathIndex, ':') ;
  ldLibraryPathVar = apt.ShellVariableAssignment('LD_LIBRARY_PATH', ldLibraryPath);
  
  % Join all the sub-commands with &&
  command0 = unsetCommand.append('&&', 'export', ldLibraryPathVar);
  if inputCommand.isNull()
    result = command0;
  else
    result = command0.cat('&&', inputCommand);
  end
else
  % Join all the sub-commands with &&
  if inputCommand.isNull()
    result = unsetCommand;
  else    
    result = unsetCommand.cat('&&', inputCommand);
  end
end

end