function testCondaBackendConfig_(obj, labeler)
  % Test the conda backend
  
  obj.testText_ = {sprintf('%s: Testing Conda Configuration...',datestr(now))};
  labeler.notify_('updateBackendTestText');

  % Check if Windows box.  Conda backend is not supported on Windows.
  obj.testText_{end+1,1} = ''; 
  labeler.notify_('updateBackendTestText');
  obj.testText_{end+1,1} = '** Checking for (lack of) Windows...'; 
  labeler.notify_('updateBackendTestText');
  if ispc(),
    obj.testText_{end+1,1} = 'FAILURE. Conda backend is not supported on Windows.'; 
    labeler.notify_('updateBackendTestText');
    return
  end
  obj.testText_{end+1,1} = 'SUCCESS!'; 
  labeler.notify_('updateBackendTestText');

  % make sure conda is installed
  obj.testText_{end+1,1} = ''; 
  labeler.notify_('updateBackendTestText');
  obj.testText_{end+1,1} = '** Checking for conda...'; 
  labeler.notify_('updateBackendTestText');
  conda_executable_path = find_conda_executable();
  condaCommand = apt.ShellCommand({conda_executable_path, '-V'}, apt.PathLocale.wsl, apt.Platform.posix);
  obj.testText_{end+1,1} = condaCommand.char(); 
  labeler.notify_('updateBackendTestText');
  [st,~] = condaCommand.run();
  if st~=0
    obj.testText_{end+1,1} = sprintf('FAILURE. Error with ''%s''. Make sure you have installed conda and added it to your PATH.',condaCommand.char()); 
    labeler.notify_('updateBackendTestText');
    return;
  end
  obj.testText_{end+1,1} = 'SUCCESS!'; 
  labeler.notify_('updateBackendTestText');

  % activate APT
  obj.testText_{end+1,1} = ''; 
  labeler.notify_('updateBackendTestText');
  obj.testText_{end+1,1} = sprintf('** Testing conda run -n %s...', obj.condaEnv); 
  labeler.notify_('updateBackendTestText');

  rawCmd = apt.ShellCommand({'echo', '"Hello, world!"'}, apt.PathLocale.wsl, apt.Platform.posix);
  command = wrapCommandConda(rawCmd, 'condaEnv', obj.condaEnv);
  obj.testText_{end+1,1} = command.char(); 
  labeler.notify_('updateBackendTestText');
  [st,~] = command.run();
  if st~=0
    obj.testText_{end+1,1} = sprintf('FAILURE. Error with ''%s''. Make sure you have created the conda environment %s',command.char(), obj.condaEnv); 
    labeler.notify_('updateBackendTestText');
    return
  end
  obj.testText_{end+1,1} = 'SUCCESS!'; 
  labeler.notify_('updateBackendTestText');
  
  % free GPUs
  obj.testText_{end+1,1} = ''; 
  labeler.notify_('updateBackendTestText');
  obj.testText_{end+1,1} = '** Looking for free GPUs ...'; 
  labeler.notify_('updateBackendTestText');
  gpuid = obj.getFreeGPUs(1,'verbose',true);
  if isempty(gpuid)
    obj.testText_{end+1,1} = 'WARNING: Could not find free GPUs. APT will run SLOWLY on CPU.'; 
    labeler.notify_('updateBackendTestText');
  else
    obj.testText_{end+1,1} = sprintf('SUCCESS! Found available GPUs.'); 
    labeler.notify_('updateBackendTestText');
  end

  obj.testText_{end+1,1} = '';
  obj.testText_{end+1,1} = 'All tests passed. Conda Backend should work for you.'; 
  labeler.notify_('updateBackendTestText');
end  % function