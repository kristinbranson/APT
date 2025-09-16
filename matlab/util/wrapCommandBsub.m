function result = wrapCommandBsub(inputCommand, varargin)
% Wrap a command for bsub job submission.  inputCommand should be a
% ShellCommand with remote locale.

% Validate input inputCommand
assert(isa(inputCommand, 'apt.ShellCommand'), 'inputCommand must be an apt.ShellCommand object');
assert(inputCommand.tfDoesMatchLocale(apt.PathLocale.wsl), 'inputCommand must have wsl locale');

[nslots,gpuqueue,logfile,jobname,additionalArgs] = ...
  myparse(varargin,...
          'nslots',DLBackEndClass.default_jrcnslots_train,...
          'gpuqueue',DLBackEndClass.default_jrcgpuqueue,...
          'logfile','/dev/null',...
          'jobname','', ...
          'additionalArgs','');
% Convert log file path to MetaPath object
logFilePathRemote = apt.MetaPath(logfile, 'wsl', 'cache');

% Build the bsub command using sequential ShellCommand objects
command0 = ...
  apt.ShellCommand({'bsub', '-n', num2str(nslots), '-gpu', 'num=1', '-q', gpuqueue, '-o', logFilePathRemote, '-R', 'affinity[core(1)]'}, ...
                   apt.PathLocale.wsl, ...
                   apt.Platform.posix);

if ~isempty(jobname)
  command1 = command0.append('-J', jobname);
else
  command1 = command0;
end

if ~isempty(additionalArgs)
  command2 = command1.append(additionalArgs);
else
  command2 = command1;
end

result = command2.append(inputCommand);  % Note that inputCommand will be a subcommand, and so will be quoted appropriately

end  % function
