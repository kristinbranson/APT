function [st,res,warningstr] = run(obj, varargin)
% Run a command using Matlab built-in system() command, but with some handy
% additional capabilities.  (This is a replacement for the apt.syscmd()
% function, which dealt in raw strings.)
%
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
% See prependStuffToClearMatlabEnvironment() if we need this to be
% fancier at some point.
doprecommand = isunix() || usewslonwindows ;
if doprecommand ,
  precommand = apt.ShellCommand({'export', 'LD_LIBRARY_PATH='}, obj.locale_, obj.platform_) ;
  command1 = apt.ShellCommand.cat(precommand, '&&', obj) ;
else
  command1 = obj ;
end

% Wrap command for running in WSL, if needed and asked for
if usewslonwindows ,
  command = wrapWslCommandForWslIfWindows(command1) ;
else
  command = command1 ;  
end

% At long last, convert the ShellCommand to a string
commandAsString = command.char() ;

% Echo the command line
if verbose ,
  fprintf('apt.ShellCommand.run(): %s\n', commandAsString);
end

% Issue the command to system()
[st,res] = system(commandAsString);

% Echo result, if called for
if verbose ,
  if st ~= 0,
    fprintf('apt.ShellCommand.run(): st = %d, res = "%s"\n',st,res);
  else
    fprintf('apt.ShellCommand.run(): success.\n');
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
      error('APT:syscmd','\nThe command:\n%s\nYielded a nonzero status code (%d):\n%s\n\n',cmdAsString,st,res);
    case 'warn'
      warningNoTrace('\nThe command:\n%s\nYielded a nonzero status code (%d):\n%s\n\n',cmdAsString,st,res);
    case 'silent'
      % do nothing
    otherwise
      error('failbehavior must be either ''err'', ''warn'', or ''silent''') ;
  end
end
