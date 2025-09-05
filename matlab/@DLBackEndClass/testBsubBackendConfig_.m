function testBsubBackendConfig_(obj, labeler)
  % Test the bsub backend
  cacheDir = labeler.DLCacheDir;
  assert(exist(cacheDir,'dir'),...
         'Deep Learning cache directory ''%s'' does not exist.',cacheDir);
  
  host = DLBackEndClass.jrchost ;

  obj.testText_ = {sprintf('%s: Testing JRC cluster backend...',datestr(now()))};
  labeler.notify('updateBackendTestText') ;

  % test that you can ping jrc host
  obj.testText_{end+1,1} = ''; 
  labeler.notify('updateBackendTestText');
  obj.testText_{end+1,1} = sprintf('** Testing that host %s can be reached...\n',host); 
  labeler.notify('updateBackendTestText')
  pingCommand = apt.ShellCommand({'ping', '-c', '1', '-W', '10', host}, apt.PathLocale.wsl, apt.Platform.posix);
  obj.testText_{end+1,1} = pingCommand.char(); 
  labeler.notify('updateBackendTestText');
  [status,result] = pingCommand.run();
  obj.testText_{end+1,1} = result; 
  labeler.notify('updateBackendTestText');
  if status ~= 0,
    obj.testText_{end+1,1} = 'FAILURE. Error with ping command.'; 
    labeler.notify('updateBackendTestText');
    return
  end
  % tried to make this robust to mac output
  m = regexp(result,' (\d+) [^,]*received','tokens','once');
  if isempty(m),
    obj.testText_{end+1,1} = 'FAILURE. Could not parse ping output.'; 
    labeler.notify('updateBackendTestText');
    return;
  end
  if str2double(m{1}) == 0,
    obj.testText_{end+1,1} = sprintf('FAILURE. Could not ping %s:\n',host); 
    labeler.notify('updateBackendTestText');
    return
  end
  obj.testText_{end+1,1} = 'SUCCESS!'; 
  labeler.notify('updateBackendTestText');

  % test that we can connect to jrc host and access CacheDir on it
  obj.testText_{end+1,1} = ''; 
  labeler.notify('updateBackendTestText');
  obj.testText_{end+1,1} = sprintf('** Testing that we can do passwordless ssh to %s...',host); 
  labeler.notify('updateBackendTestText');
  touchPathAsCharray = fullfile(cacheDir,sprintf('testBsub_test_%s.txt',datestr(now(),'yyyymmddTHHMMSS.FFF')));
  touchFilePath = apt.MetaPath(touchPathAsCharray, apt.PathLocale.remote, apt.FileRole.cache);

  baseTouchCommand = ...
    apt.ShellCommand({'touch', touchFilePath, ';', 'if', '[', '-e', touchFilePath, '];', ...
                      'then', 'rm', '-f', touchFilePath, '&&', 'echo', '"SUCCESS"', ';', 'else', 'echo', '"FAILURE"', ';', 'fi', ';'}, ...
                     apt.PathLocale.remote, apt.Platform.posix);
  timeout = 20;
  touchCommand = wrapCommandSSH(baseTouchCommand,'host',host,'timeout',timeout);
  obj.testText_{end+1,1} = touchCommand.char(); 
  labeler.notify('updateBackendTestText');
  [status,result] = touchCommand.run();
  obj.testText_{end+1,1} = result; 
  labeler.notify('updateBackendTestText');
  if status ~= 0,
    obj.testText_{end+1,1} = ...
      sprintf('ssh command timed out. This could be because passwordless ssh to %s has not been set up. Please see APT wiki for more details.',host);
    labeler.notify('updateBackendTestText');
    return;
  end
  issuccess = contains(result,'SUCCESS');
  isfailure = contains(result,'FAILURE');
  if issuccess && ~isfailure,
    obj.testText_{end+1,1} = 'SUCCESS!'; 
    labeler.notify('updateBackendTestText');
  elseif ~issuccess && isfailure,
    obj.testText_{end+1,1} = sprintf('FAILURE. Could not create file in CacheDir %s:',cacheDir); 
    labeler.notify('updateBackendTestText');
    return
  else
    obj.testText_{end+1,1} = 'FAILURE. ssh test failed.'; 
    labeler.notify('updateBackendTestText');
    return
  end

  % test that we can run bjobs
  obj.testText_{end+1,1} = '** Testing that we can interact with the cluster...'; 
  labeler.notify('updateBackendTestText');
  baseBjobsCommand = apt.ShellCommand({'bjobs'}, apt.PathLocale.remote, apt.Platform.posix);
  bjobsCommand = wrapCommandSSH(baseBjobsCommand,'host',host);
  obj.testText_{end+1,1} = bjobsCommand.char(); labeler.notify('updateBackendTestText');
  [status,result] = bjobsCommand.run();
  obj.testText_{end+1,1} = result; labeler.notify('updateBackendTestText');
  if status ~= 0,
    obj.testText_{end+1,1} = sprintf('Error running bjobs on %s',host); 
    labeler.notify('updateBackendTestText');
    return
  end
  obj.testText_{end+1,1} = 'SUCCESS!';
  obj.testText_{end+1,1} = '';
  obj.testText_{end+1,1} = 'All tests passed. JRC Backend should work for you.'; 
  labeler.notify('updateBackendTestText');
end  % function
