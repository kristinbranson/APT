function [tfsucc,res,warningstr,st] = syscmd(cmd0, varargin)
% Run a command using Matlab built-in system() command, but with some handy
% additional capabilities.
% Optional args allow caller to specify what to do if the command fails.  (Can
% error, warn, or silently ignore.)  Also has option to lightly process a JSON
% response.

% Process keyword args
[failbehavior,isjsonout,setenvcmd,usewslonwindows] = ...
  myparse(varargin,...
          'failbehavior','warn',... % one of 'err','warn','silent'
          'isjsonout',false,...
          'setenvcmd','LD_LIBRARY_PATH=',...
          'usewslonwindows',false);
  % Default setenvcmd is to prevent Matlab's very matlab-specific
  % LD_LIBRARY_PATH from messing up normal commands.
  % See prepend_stuff_to_clear_matlab_environment() if we need this to be 
  % fancier at some point.    

% Prepend the LD_LIBRARY_PATH bit if on Linux or WSL  
dosetenv = isunix() || usewslonwindows ;
if dosetenv && ~isempty(setenvcmd) ,
  cmd1 = sprintf('%s %s', setenvcmd, cmd0) ;
else
  cmd1 = cmd0 ;
end

% Wrap command for running in WSL, if needed and asked for
if usewslonwindows ,
  cmd = wrap_linux_command_line_for_wsl_if_windows(cmd1) ;
else
  cmd = cmd1 ;  
end

% Echo, & run
fprintf('apt.syscmd(): %s\n',cmd);
[st,res] = system(cmd);

% Echo result, decide if it was successful
if st ~= 0,
  fprintf('st = %d, res = "%s"\n',st,res);
else
  fprintf('success.\n');
end
%tfsucc = st==0 || isempty(res);  % Why does isempty(res) count as success?  --ALT, 2024-09-12
tfsucc = (st==0) ;

% Parse output JSON, if requested
if isjsonout && tfsucc ,
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

% If failed, do the appropriate thing
if ~tfsucc 
  switch failbehavior
    case 'err'
      error('The command:\n%s\nYielded a nonzero status code (%d):\n%s\n',cmd,st,res);
    case 'warn'
      warningNoTrace('The command:\m%s\nYielded a nonzero status code (%d):\n%s\n',cmd,st,res);
    case 'silent'
      % do nothing
    otherwise
      error('failbehavior must be either ''err'', ''warn'', or ''silent''') ;
  end
end
