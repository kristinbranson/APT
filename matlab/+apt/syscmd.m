function [st,res,warningstr] = syscmd(cmd0, varargin)
% Run a command using Matlab built-in system() command, but with some handy
% additional capabilities.
% Optional args allow caller to specify what to do if the command fails.  (Can
% error, warn, or silently ignore.)  Also has option to lightly process a JSON
% response.

% Process keyword args
[failbehavior,isjsonout,usewslonwindows,verbose] = ...
  myparse(varargin,...
          'failbehavior','warn',... % one of 'err','warn','silent'
          'isjsonout',false,...
          'usewslonwindows',true,...
          'verbose',true);

% Prepend the LD_LIBRARY_PATH bit if on Linux or WSL  
% precommand is to prevent Matlab's very matlab-specific
% LD_LIBRARY_PATH from messing up normal commands.
% See prepend_stuff_to_clear_matlab_environment() if we need this to be
% fancier at some point.
doprecommand = isunix() || usewslonwindows ;
if doprecommand ,
  precommand = 'export LD_LIBRARY_PATH=' ;
  cmd1 = sprintf('%s && %s', precommand, cmd0) ;
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
if verbose ,
  fprintf('apt.syscmd(): %s\n',cmd);
end
[st,res] = system(cmd);

% Echo result, if called for
if verbose ,
  if st ~= 0,
    fprintf('apt.syscmd(): st = %d, res = "%s"\n',st,res);
  else
    fprintf('apt.syscmd(): success.\n');
  end
end

% Parse output JSON, if requested
if isjsonout && (st==0) ,
  jsonstart = find(res == '{',1);
  if isempty(jsonstart),
    st = 0 ;
    warningstr = 'Could not find json start character {';
  else
    warningstr = res(1:jsonstart-1);
    res = res(jsonstart:end);
  end
else
  warningstr = '';
end

% If failed, do the appropriate thing
if st ~= 0 , 
  switch failbehavior
    case 'err'
      error('APT:syscmd','\nThe command:\n%s\nYielded a nonzero status code (%d):\n%s\n\n',cmd,st,res);
    case 'warn'
      warningNoTrace('\nThe command:\n%s\nYielded a nonzero status code (%d):\n%s\n\n',cmd,st,res);
    case 'silent'
      % do nothing
    otherwise
      error('failbehavior must be either ''err'', ''warn'', or ''silent''') ;
  end
end
