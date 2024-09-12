function [tfsucc,res,warningstr] = syscmd(cmd,varargin)
% Run a command using Matlab built-in system() command, but with some handy
% additional capabilities.
% Optional args allow caller to specify what to do if the command fails.  (Can
% error, warn, or silently ignore.)  Also has option to lightly process a JSON
% response.
%
% Seems to also support issuing the command using the Java runtime, but AFAICT
% this is never used in APT.  -- ALT, 2024-04-25

% Process keyword args
[dispcmd,failbehavior,isjsonout,dosetenv,setenvcmd,usejavaRT,dosleep] = ...
  myparse(varargin,...
          'dispcmd',false,...
          'failbehavior','warn',... % one of 'err','warn','silent'
          'isjsonout',false,...
          'dosetenv',isunix(),...
          'setenvcmd','',...
          'usejavaRT',false, ...
          'dosleep', true);
      
if isempty(setenvcmd) ,
  if dosleep ,
    setenvcmd = 'sleep 5 ; LD_LIBRARY_PATH= AWS_PAGER= ' ;
      % Change the sleep number at your own risk!  I tried 3, and everything seemed
      % fine for a while until it became a very hard-to-find bug. -- ALT, 2024-06-28
  else
    setenvcmd = 'LD_LIBRARY_PATH= AWS_PAGER= ' ;
  end          
end

if dosetenv,
  cmd = sprintf('%s %s', setenvcmd, cmd) ;
end

% XXX HACK
drawnow 

if dispcmd
  disp(cmd); 
end
if usejavaRT
  fprintf(1,'Using javaRT call\n');
  runtime = java.lang.Runtime.getRuntime();
  proc = runtime.exec(cmd);
  st = proc.waitFor();
  is = proc.getInputStream;
  res = [];
  val = is.read();
  while val~=-1 && numel(res)<100
    res(1,end+1) = val;  %#ok<AGROW> 
    val = is.read();
  end
  res = strtrim(char(res));
  tfsucc = st==0;
else
  fprintf('syscmd: %s\n',cmd);
  [st,res] = system(cmd);
  if st ~= 0,
    fprintf('st = %d, res = %s\n',st,res);
  else
    fprintf('success.\n');
  end
  tfsucc = st==0 || isempty(res);
end

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
      error('Nonzero status code: %s',res);
    case 'warn'
      warningNoTrace('Command failed: %s: %s',cmd,res);
    case 'silent'
      % none
    otherwise
      assert(false);
  end
end
