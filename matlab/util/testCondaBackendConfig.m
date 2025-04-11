function [tfsucc,hedit] = testCondaBackendConfig(backend)
tfsucc = false;

[~,hedit] = createFigTestBackendConfig('Test Conda Configuration');      
hedit.String = {sprintf('%s: Testing Conda Configuration...',datestr(now))}; 
drawnow;

% Check if Windows box.  Conda backend is not supported on Windows.
hedit.String{end+1} = ''; drawnow;
hedit.String{end+1} = '** Checking for (lack of) Windows...'; drawnow;
if ispc() ,
  hedit.String{end+1} = 'FAILURE. Conda backend is not supported on Windows.'; drawnow;
  return
end
hedit.String{end+1} = 'SUCCESS!'; drawnow;

% make sure conda is installed
hedit.String{end+1} = ''; drawnow;
hedit.String{end+1} = '** Checking for conda...'; drawnow;
conda_executable_path = find_conda_executable() ;
cmd = sprintf('%s -V', conda_executable_path) ;
hedit.String{end+1} = cmd; drawnow;
[st,~] = apt.syscmd(cmd);
%reslines = splitlines(res);
if st~=0
  hedit.String{end+1} = sprintf('FAILURE. Error with ''%s''. Make sure you have installed conda and added it to your PATH.',cmd); drawnow;
  return;
end
hedit.String{end+1} = 'SUCCESS!'; drawnow;


% activate APT
hedit.String{end+1} = ''; drawnow;
hedit.String{end+1} = sprintf('** Testing conda run -n %s...', backend.condaEnv); 
drawnow;

raw_cmd = 'echo "Hello, world!"' ;
cmd = wrapCommandConda(raw_cmd, 'condaEnv', backend.condaEnv) ;
%fprintf(1,'%s\n',cmd);
hedit.String{end+1} = cmd; drawnow;
[st,~] = apt.syscmd(cmd);
%reslines = splitlines(res);
%reslinesdisp = reslines(1:min(4,end));
%hedit.String = [hedit.String; reslinesdisp(:)];
if st~=0
  hedit.String{end+1} = sprintf('FAILURE. Error with ''%s''. Make sure you have created the conda environment %s',cmd, backend.condaEnv); 
  drawnow;
  return
end
hedit.String{end+1} = 'SUCCESS!'; drawnow;
  
% free GPUs
hedit.String{end+1} = ''; drawnow;
hedit.String{end+1} = '** Looking for free GPUs ...'; drawnow;
gpuid = backend.getFreeGPUs(1,'verbose',true);
if isempty(gpuid)
  hedit.String{end+1} = 'WARNING: Could not find free GPUs. APT will run SLOWLY on CPU.'; drawnow;
else
  hedit.String{end+1} = sprintf('SUCCESS! Found available GPUs.'); drawnow;
end

hedit.String{end+1} = '';
hedit.String{end+1} = 'All tests passed. Conda Backend should work for you.'; drawnow;

tfsucc = true;      
