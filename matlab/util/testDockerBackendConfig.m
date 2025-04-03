function [tfsucc,hedit] = testDockerBackendConfig(backend)

tfsucc = false;

[~,hedit] = createFigTestBackendConfig('Test Docker Configuration');      
hedit.String = {sprintf('%s: Testing Docker Configuration...',datestr(now))}; 
drawnow;

% docker hello world
hedit.String{end+1} = ''; drawnow;
hedit.String{end+1} = '** Testing docker hello-world...'; drawnow;

dockercmd = apt.dockercmd();
cmd = sprintf('%s run --rm hello-world',dockercmd);

if ~isempty(backend.dockerremotehost),
  cmd = wrapCommandSSH(cmd,'host',backend.dockerremotehost);
end

fprintf(1,'%s\n',cmd);
hedit.String{end+1} = cmd; 
drawnow;
[st,res] = apt.syscmd(cmd) ;
reslines = splitlines(res);
reslinesdisp = reslines(1:min(4,end));
hedit.String = [hedit.String; reslinesdisp(:)];
if st~=0
  hedit.String{end+1} = 'FAILURE. Error with docker run command.'; drawnow;
  return;
end
hedit.String{end+1} = 'SUCCESS!'; drawnow;

% docker (api) version
hedit.String{end+1} = ''; drawnow;
hedit.String{end+1} = '** Checking docker API version...'; drawnow;

[tfsucc,~,clientapiver] = backend.getDockerVers();
if ~tfsucc        
  hedit.String{end+1} = 'FAILURE. Failed to ascertain docker API version.'; drawnow;
  return;
end

tfsucc = false;
% In this conditional we assume the apiver numbering scheme continues
% like '1.39', '1.40', ... 
if ~(str2double(clientapiver)>=str2double(apt.docker_api_version()))          
  hedit.String{end+1} = ...
    sprintf('FAILURE. Docker API version %s does not meet required minimum of %s.',...
      clientapiver,apt.docker_api_version());
  drawnow;
  return;
end        
succstr = sprintf('SUCCESS! Your Docker API version is %s.',clientapiver);
hedit.String{end+1} = succstr; drawnow;      

% APT hello
hedit.String{end+1} = ''; drawnow;
hedit.String{end+1} = '** Testing APT deepnet library...'; 
hedit.String{end+1} = '   (This can take some time the first time the docker image is pulled)'; 
drawnow;
deepnetroot = [APT.Root '/deepnet'];
homedir = get_home_dir_name() ;
%deepnetrootguard = [filequote deepnetroot filequote];
basecmd = 'python APT_interface.py lbl test hello';
cmd = wrapCommandDocker(basecmd,...
                        'dockerimg', backend.dockerimgfull, ...
                        'containername','containerTest',...
                        'detach',false,...
                        'bindpath',{wsl_path_from_native(deepnetroot),wsl_path_from_native(homedir)});
hedit.String{end+1} = cmd;
RUNAPTHELLO = 1;
if RUNAPTHELLO % AL: this may not work property on a multi-GPU machine with some GPUs in use
  %fprintf(1,'%s\n',cmd);
  %hedit.String{end+1} = cmd; drawnow;
  [st,res] = apt.syscmd(cmd);
  reslines = splitlines(res);
  reslinesdisp = reslines(1:min(4,end));
  hedit.String = [hedit.String; reslinesdisp(:)];
  if st~=0
    hedit.String{end+1} = 'FAILURE. Error with APT deepnet command.'; drawnow;
    return;
  end
  hedit.String{end+1} = 'SUCCESS!'; drawnow;
end

% free GPUs
hedit.String{end+1} = ''; drawnow;
hedit.String{end+1} = '** Looking for free GPUs ...'; drawnow;
[gpuid,~,~] = backend.getFreeGPUs(1,'verbose',true);
if isempty(gpuid)
  hedit.String{end+1} = 'WARNING. Could not find free GPUs. APT will run SLOWLY on CPU.'; drawnow;
else
  hedit.String{end+1} = 'SUCCESS! Found available GPUs.'; drawnow;
end

hedit.String{end+1} = '';
hedit.String{end+1} = 'All tests passed. Docker Backend should work for you.'; drawnow;

tfsucc = true;      
