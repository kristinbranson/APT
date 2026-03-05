function result = wrapCommandBsub(inputCommand, varargin)
% Wrap a command for bsub job submission.  inputCommand should be a
% ShellCommand with remote locale.

% Validate inputCommand
assert(isa(inputCommand, 'apt.ShellCommand'), 'inputCommand must be an apt.ShellCommand object');
assert(inputCommand.tfDoesMatchLocale(apt.PathLocale.wsl), 'inputCommand must have wsl locale');

% Deal with optional arguments
[nslots,gpuqueue,logfile,jobname,additionalArgs,jobDuration] = ...
  myparse(varargin,...
          'nslots',DLBackEndClass.default_jrcnslots_train,...
          'gpuqueue',DLBackEndClass.default_jrcgpuqueue,...
          'logfile','/dev/null',...
          'jobname','', ...
          'additionalArgs','',...
		      'jobDuration',2880);

% Convert log file path to MetaPath object
logFilePathRemote = apt.MetaPath(logfile, 'wsl', 'cache');

% Build the bsub command using sequential ShellCommand objects
command0 = ...
  apt.ShellCommand({'bsub', '-n', num2str(nslots), '-gpu', 'num=1', '-W', jobDuration, '-q', gpuqueue, '-o', logFilePathRemote, ...
                    '-R', escape_string_for_bash('affinity[core(1)]')}, ...
                   apt.PathLocale.wsl, ...
                   apt.Platform.posix);

% Append job name, if given
if ~isempty(jobname)
  command1 = command0.append('-J', jobname);
else
  command1 = command0;
end

% Append additional args, if given
if ~isempty(additionalArgs)
  command2 = command1.append(additionalArgs);
else
  command2 = command1;
end

% Bring it all together
result = command2.append(inputCommand);  % Note that inputCommand will be a subcommand, and so will be quoted appropriately

end  % function
