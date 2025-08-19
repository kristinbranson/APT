function testDockerBackendConfig_(obj, labeler)
  % Test the docker backend

  obj.testText_ = {sprintf('%s: Testing Docker Configuration...',datestr(now))};
  labeler.notify('updateBackendTestText');

  % docker hello world
  obj.testText_{end+1} = ''; 
  labeler.notify('updateBackendTestText');
  obj.testText_{end+1} = '** Testing docker hello-world...'; 
  labeler.notify('updateBackendTestText');

  dockercmd = apt.dockercmd();
  cmd = sprintf('%s run --rm hello-world',dockercmd);

  if ~isempty(obj.dockerremotehost),
    cmd = wrapCommandSSH(cmd,'host',obj.dockerremotehost);
  end

  fprintf(1,'%s\n',cmd);
  obj.testText_{end+1} = cmd; 
  labeler.notify('updateBackendTestText');
  [st,res] = apt.syscmd(cmd);
  reslines = splitlines(res);
  reslinesdisp = reslines(1:min(4,end));
  obj.testText_ = [obj.testText_; reslinesdisp(:)];
  labeler.notify('updateBackendTestText');
  if st~=0
    obj.testText_{end+1} = 'FAILURE. Error with docker run command.'; 
    labeler.notify('updateBackendTestText');
    return;
  end
  obj.testText_{end+1} = 'SUCCESS!'; 
  labeler.notify('updateBackendTestText');

  % docker (api) version
  obj.testText_{end+1} = ''; 
  labeler.notify('updateBackendTestText');
  obj.testText_{end+1} = '** Checking docker API version...'; 
  labeler.notify('updateBackendTestText');

  [tfsucc,~,clientapiver] = obj.getDockerVers();
  if ~tfsucc        
    obj.testText_{end+1} = 'FAILURE. Failed to ascertain docker API version.'; 
    labeler.notify('updateBackendTestText');
    return;
  end

  % In this conditional we assume the apiver numbering scheme continues
  % like '1.39', '1.40', ... 
  if ~(str2double(clientapiver)>=str2double(apt.docker_api_version()))          
    obj.testText_{end+1} = ...
      sprintf('FAILURE. Docker API version %s does not meet required minimum of %s.',...
        clientapiver,apt.docker_api_version());
    labeler.notify('updateBackendTestText');
    return;
  end        
  succstr = sprintf('SUCCESS! Your Docker API version is %s.',clientapiver);
  obj.testText_{end+1} = succstr; 
  labeler.notify('updateBackendTestText');      

  % APT hello
  obj.testText_{end+1} = ''; 
  labeler.notify('updateBackendTestText');
  obj.testText_{end+1} = '** Testing APT deepnet library...'; 
  labeler.notify('updateBackendTestText');
  obj.testText_{end+1} = '   (This can take some time the first time the docker image is pulled)'; 
  labeler.notify('updateBackendTestText');
  deepnetroot = [APT.Root '/deepnet'];
  homedir = get_home_dir_name();
  basecmd = 'python APT_interface.py lbl test hello';
  cmd = wrapCommandDocker(basecmd,...
                          'dockerimg', obj.dockerimgfull, ...
                          'containername','containerTest',...
                          'detach',false,...
                          'bindpath',{wsl_path_from_native(deepnetroot),wsl_path_from_native(homedir)});
  obj.testText_{end+1} = cmd;
  labeler.notify('updateBackendTestText');
  RUNAPTHELLO = 1;
  if RUNAPTHELLO % AL: this may not work property on a multi-GPU machine with some GPUs in use
    [st,res] = apt.syscmd(cmd);
    reslines = splitlines(res);
    reslinesdisp = reslines(1:min(4,end));
    obj.testText_ = [obj.testText_; reslinesdisp(:)];
    labeler.notify('updateBackendTestText');
    if st~=0
      obj.testText_{end+1} = 'FAILURE. Error with APT deepnet command.'; 
      labeler.notify('updateBackendTestText');
      return;
    end
    obj.testText_{end+1} = 'SUCCESS!'; 
    labeler.notify('updateBackendTestText');
  end

  % free GPUs
  obj.testText_{end+1} = ''; 
  labeler.notify('updateBackendTestText');
  obj.testText_{end+1} = '** Looking for free GPUs ...'; 
  labeler.notify('updateBackendTestText');
  [gpuid,~,~] = obj.getFreeGPUs(1,'verbose',true);
  if isempty(gpuid)
    obj.testText_{end+1} = 'WARNING. Could not find free GPUs. APT will run SLOWLY on CPU.'; 
    labeler.notify('updateBackendTestText');
  else
    obj.testText_{end+1} = 'SUCCESS! Found available GPUs.'; 
    labeler.notify('updateBackendTestText');
  end

  obj.testText_{end+1} = '';
  obj.testText_{end+1} = 'All tests passed. Docker Backend should work for you.'; 
  labeler.notify('updateBackendTestText');
end  % function