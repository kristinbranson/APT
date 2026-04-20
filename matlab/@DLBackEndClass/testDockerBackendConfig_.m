function testDockerBackendConfig_(obj, labeler)
  % Test the docker backend

  obj.testText_ = {sprintf('%s: Testing Docker Configuration...',datestr(now))};
  labeler.notifyRetrograde('updateBackendTestText');

  % docker hello world
  obj.testText_{end+1,1} = ''; 
  labeler.notifyRetrograde('updateBackendTestText');
  obj.testText_{end+1,1} = '** Testing docker hello-world...'; 
  labeler.notifyRetrograde('updateBackendTestText');

  dockercmd = apt.dockercmd();
  command = apt.ShellCommand({dockercmd, 'run', '--rm', 'hello-world'}, apt.PathLocale.wsl, apt.Platform.posix);

  if ~isempty(obj.dockerremotehost),
    command = wrapCommandSSH(command,'host',obj.dockerremotehost);
  end

  fprintf(1,'%s\n',command.char());
  obj.testText_{end+1,1} = command.char(); 
  labeler.notifyRetrograde('updateBackendTestText');
  [st,res] = command.run();
  reslines = splitlines(res);
  reslinesdisp = reslines(1:min(4,end));
  obj.testText_ = [obj.testText_; reslinesdisp(:)];
  labeler.notifyRetrograde('updateBackendTestText');
  if st~=0
    obj.testText_{end+1,1} = 'FAILURE. Error with docker run command.'; 
    labeler.notifyRetrograde('updateBackendTestText');
    return;
  end
  obj.testText_{end+1,1} = 'SUCCESS!'; 
  labeler.notifyRetrograde('updateBackendTestText');

  % docker (api) version
  obj.testText_{end+1,1} = ''; 
  labeler.notifyRetrograde('updateBackendTestText');
  obj.testText_{end+1,1} = '** Checking docker API version...'; 
  labeler.notifyRetrograde('updateBackendTestText');

  [tfsucc,~,clientapiver] = obj.getDockerVers();
  if ~tfsucc        
    obj.testText_{end+1,1} = 'FAILURE. Failed to ascertain docker API version.'; 
    labeler.notifyRetrograde('updateBackendTestText');
    return;
  end

  % In this conditional we assume the apiver numbering scheme continues
  % like '1.39', '1.40', ... 
  % if ~(str2double(clientapiver)>=str2double(apt.docker_api_version()))          
  %   obj.testText_{end+1,1} = ...
  %     sprintf('FAILURE. Docker API version %s does not meet required minimum of %s.',...
  %       clientapiver,apt.docker_api_version());
  %   labeler.notifyRetrograde('updateBackendTestText');
  %   return;
  % end        
  succstr = sprintf('SUCCESS! Your Docker API version is %s.',clientapiver);
  obj.testText_{end+1,1} = succstr; 
  labeler.notifyRetrograde('updateBackendTestText');      

  % APT hello
  obj.testText_{end+1,1} = ''; 
  labeler.notifyRetrograde('updateBackendTestText');
  obj.testText_{end+1,1} = '** Testing APT deepnet library...'; 
  labeler.notifyRetrograde('updateBackendTestText');
  obj.testText_{end+1,1} = '   (This can take some time the first time the docker image is pulled)'; 
  labeler.notifyRetrograde('updateBackendTestText');
  deepnetRootNativeAsChar = fullfile(APT.Root, 'deepnet');
  homedir = get_home_dir_name();
  deepnetrootPath = apt.MetaPath(deepnetRootNativeAsChar, 'native', 'source');
  homedirPath = apt.MetaPath(homedir, 'native', 'home');
  baseCommand = apt.ShellCommand({'python', 'APT_interface.py', 'lbl', 'test', 'hello'}, apt.PathLocale.wsl, apt.Platform.posix);
  command = wrapCommandDocker(baseCommand,...
                              'dockerimg', obj.dockerimgfull, ...
                              'containername','containerTest',...
                              'detach',false,...
                              'bindpath',{deepnetrootPath, homedirPath});
  obj.testText_{end+1,1} = command.char();
  labeler.notifyRetrograde('updateBackendTestText');
  RUNAPTHELLO = 1;
  if RUNAPTHELLO % AL: this may not work property on a multi-GPU machine with some GPUs in use
    [st,res] = command.run();
    reslines = splitlines(res);
    reslinesdisp = reslines(1:min(4,end));
    obj.testText_ = [obj.testText_; reslinesdisp(:)];
    labeler.notifyRetrograde('updateBackendTestText');
    if st~=0
      obj.testText_{end+1,1} = 'FAILURE. Error with APT deepnet command.'; 
      labeler.notifyRetrograde('updateBackendTestText');
      return;
    end
    obj.testText_{end+1,1} = 'SUCCESS!'; 
    labeler.notifyRetrograde('updateBackendTestText');
  end

  % free GPUs
  obj.testText_{end+1,1} = ''; 
  labeler.notifyRetrograde('updateBackendTestText');
  obj.testText_{end+1,1} = '** Looking for free GPUs ...'; 
  labeler.notifyRetrograde('updateBackendTestText');
  [gpuid,~,~] = obj.getFreeGPUs(1,'verbose',true);
  if isempty(gpuid)
    obj.testText_{end+1,1} = 'WARNING. Could not find free GPUs. APT will run SLOWLY on CPU.'; 
    labeler.notifyRetrograde('updateBackendTestText');
  else
    obj.testText_{end+1,1} = 'SUCCESS! Found available GPUs.'; 
    labeler.notifyRetrograde('updateBackendTestText');
  end

  obj.testText_{end+1,1} = '';
  obj.testText_{end+1,1} = 'All tests passed. Docker Backend should work for you.'; 
  labeler.notifyRetrograde('updateBackendTestText');
end  % function