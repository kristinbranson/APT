function [tfsucc,res,warningstr] = syscmd(cmd,varargin)
% Run a command using Matlab built-in system() command, but with some handy
% additional capabilities.
% Optional args allow caller to specify what to do if the command fails.  (Can
% error, warn, or silently ignore.)  Also has option to lightly process a JSON
% response.

% Process keyword args
[failbehavior,isjsonout,setenvcmd] = ...
  myparse(varargin,...
          'failbehavior','warn',... % one of 'err','warn','silent'
          'isjsonout',false,...
          'setenvcmd','LD_LIBRARY_PATH=');
  % Default setenvcmd is to prevent Matlab's very matlab-specific
  % LD_LIBRARY_PATH from messing up normal commands.
      
dosetenv = isunix() ;
if dosetenv && ~isempty(setenvcmd) ,
  cmd = sprintf('%s %s', setenvcmd, cmd) ;
end

fprintf('apt.syscmd(): %s\n',cmd);
[st,res] = system(cmd);
if st ~= 0,
  fprintf('st = %d, res = %s\n',st,res);
else
  fprintf('success.\n');
end
tfsucc = st==0 || isempty(res);  % Why does isempty(res) count as success?  --ALT, 2024-09-12

if isjsonout && tfsucc,
  jsonstart = find(res == '{',1);
  if isempty(jsonstart),
    tfsucc = false;
    warningstr = 'Could not find json start character {';
  else
    warningstr = res(1:jsonstart-1);
    res = res(jsonstart:end);
  end
else
  warningstr = '';
end

if ~tfsucc 
  switch failbehavior
    case 'err'
      error('Nonzero status code (%d): %s',st,res);
    case 'warn'
      warningNoTrace('Command failed (status code %d): %s: %s',st,cmd,res);
    case 'silent'
      % none
    otherwise
      assert(false);
  end
end
