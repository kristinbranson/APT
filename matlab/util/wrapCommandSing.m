function result = wrapCommandSing(inputCommand, varargin)
% Wrap a command to run in Sinularity/Apptainer.  inputCommand should be a
% ShellCommand with locale wsl.

% Validate input inputCommand
assert(isa(inputCommand, 'apt.ShellCommand'), 'inputCommand must be an apt.ShellCommand object');
assert(inputCommand.tfDoesMatchLocale(apt.PathLocale.wsl), 'inputCommand must have WSL locale');

DEFAULT_BIND_PATHS_ASCHAR = {
  '/groups'
  '/nrs'
  '/scratch'};
[bindpath,singularityImagePath] = ...
  myparse(varargin,...
          'bindpath',DEFAULT_BIND_PATHS_ASCHAR,...
          'singimg','');
assert(~isempty(singularityImagePath)) ;

% Convert bind paths to MetaPath objects and create bind arguments
bindPathsRemote = cellfun(@(x) apt.MetaPath(x.path, 'wsl', 'universal'), bindpath, 'UniformOutput', false);
bindArgs = apt.ShellCommand({}, apt.PathLocale.wsl, apt.Platform.posix);
for i = 1:numel(bindPathsRemote)
  bindArgs = bindArgs.append('-B', bindPathsRemote{i});
end

% Build the final singularity command using sequential ShellCommand objects
command0 = apt.ShellCommand({'singularity', 'exec', '--nv'}, apt.PathLocale.wsl, apt.Platform.posix);
command1 = command0.cat(bindArgs);
command2 = command1.append(singularityImagePath, 'bash', '-c');
result = command2.append(inputCommand) ;
