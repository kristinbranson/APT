function testCondaBackendConfig_(obj, labeler)
  % Test the conda backend
  
  obj.testText_ = {sprintf('%s: Testing Conda Configuration...',datestr(now))};
  labeler.notify('updateBackendTestText');

  % Check if Windows box.  Conda backend is not supported on Windows.
  obj.testText_{end+1,1} = ''; 
  labeler.notify('updateBackendTestText');
  obj.testText_{end+1,1} = '** Checking for (lack of) Windows...'; 
  labeler.notify('updateBackendTestText');
  if ispc(),
    obj.testText_{end+1,1} = 'FAILURE. Conda backend is not supported on Windows.'; 
    labeler.notify('updateBackendTestText');
    return
  end
  obj.testText_{end+1,1} = 'SUCCESS!'; 
  labeler.notify('updateBackendTestText');

  % make sure conda is installed
  obj.testText_{end+1,1} = ''; 
  labeler.notify('updateBackendTestText');
  obj.testText_{end+1,1} = '** Checking for conda...'; 
  labeler.notify('updateBackendTestText');
  conda_executable_path = find_conda_executable();
  cmd = sprintf('%s -V', conda_executable_path);
  obj.testText_{end+1,1} = cmd; 
  labeler.notify('updateBackendTestText');
  [st,~] = apt.syscmd(cmd);
  if st~=0
    obj.testText_{end+1,1} = sprintf('FAILURE. Error with ''%s''. Make sure you have installed conda and added it to your PATH.',cmd); 
    labeler.notify('updateBackendTestText');
    return;
  end
  obj.testText_{end+1,1} = 'SUCCESS!'; 
  labeler.notify('updateBackendTestText');

  % activate APT
  obj.testText_{end+1,1} = ''; 
  labeler.notify('updateBackendTestText');
  obj.testText_{end+1,1} = sprintf('** Testing conda run -n %s...', obj.condaEnv); 
  labeler.notify('updateBackendTestText');

  raw_cmd = 'echo "Hello, world!"';
  cmd = wrapCommandConda(raw_cmd, 'condaEnv', obj.condaEnv);
  obj.testText_{end+1,1} = cmd; 
  labeler.notify('updateBackendTestText');
  [st,~] = apt.syscmd(cmd);
  if st~=0
    obj.testText_{end+1,1} = sprintf('FAILURE. Error with ''%s''. Make sure you have created the conda environment %s',cmd, obj.condaEnv); 
    labeler.notify('updateBackendTestText');
    return
  end
  obj.testText_{end+1,1} = 'SUCCESS!'; 
  labeler.notify('updateBackendTestText');
  
  % free GPUs
  obj.testText_{end+1,1} = ''; 
  labeler.notify('updateBackendTestText');
  obj.testText_{end+1,1} = '** Looking for free GPUs ...'; 
  labeler.notify('updateBackendTestText');
  gpuid = obj.getFreeGPUs(1,'verbose',true);
  if isempty(gpuid)
    obj.testText_{end+1,1} = 'WARNING: Could not find free GPUs. APT will run SLOWLY on CPU.'; 
    labeler.notify('updateBackendTestText');
  else
    obj.testText_{end+1,1} = sprintf('SUCCESS! Found available GPUs.'); 
    labeler.notify('updateBackendTestText');
  end

  obj.testText_{end+1,1} = '';
  obj.testText_{end+1,1} = 'All tests passed. Conda Backend should work for you.'; 
  labeler.notify('updateBackendTestText');
end  % function