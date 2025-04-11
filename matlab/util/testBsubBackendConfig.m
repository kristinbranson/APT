function [tfsucc,hedit] = testBsubBackendConfig(backend,cacheDir)  %#ok<INUSD> 
tfsucc = false;
host = DLBackEndClass.jrchost ;

[~,hedit] = createFigTestBackendConfig('Test JRC Cluster Backend');
hedit.String = {sprintf('%s: Testing JRC cluster backend...',datestr(now()))};
drawnow;

% test that you can ping jrc host
hedit.String{end+1} = ''; drawnow;
hedit.String{end+1} = sprintf('** Testing that host %s can be reached...\n',host); drawnow;
cmd = sprintf('ping -c 1 -W 10 %s',host);
hedit.String{end+1} = cmd; drawnow;
[status,result] = apt.syscmd(cmd);
hedit.String{end+1} = result; drawnow;
if status ~= 0,
  hedit.String{end+1} = 'FAILURE. Error with ping command.'; drawnow;
  return;
end
% tried to make this robust to mac output
m = regexp(result,' (\d+) [^,]*received','tokens','once');
if isempty(m),
  hedit.String{end+1} = 'FAILURE. Could not parse ping output.'; drawnow;
  return;
end
if str2double(m{1}) == 0,
  hedit.String{end+1} = sprintf('FAILURE. Could not ping %s:\n',host); drawnow;
  return;
end
hedit.String{end+1} = 'SUCCESS!'; drawnow;

% test that we can connect to jrc host and access CacheDir on it

hedit.String{end+1} = ''; drawnow;
hedit.String{end+1} = sprintf('** Testing that we can do passwordless ssh to %s...',host); drawnow;
touchfile = fullfile(cacheDir,sprintf('testBsub_test_%s.txt',datestr(now,'yyyymmddTHHMMSS.FFF')));

remotecmd = sprintf('touch "%s"; if [ -e "%s" ]; then rm -f "%s" && echo "SUCCESS"; else echo "FAILURE"; fi;',touchfile,touchfile,touchfile);
timeout = 20;
cmd1 = wrapCommandSSH(remotecmd,'host',host,'timeout',timeout);
%cmd = sprintf('timeout 20 %s',cmd1);
cmd = cmd1;
hedit.String{end+1} = cmd; drawnow;
[status,result] = apt.syscmd(cmd);
hedit.String{end+1} = result; drawnow;
if status ~= 0,
  hedit.String{end+1} = ...
    sprintf('ssh command timed out. This could be because passwordless ssh to %s has not been set up. Please see APT wiki for more details.',host); 
  drawnow;
  return;
end
issuccess = contains(result,'SUCCESS');
isfailure = contains(result,'FAILURE');
if issuccess && ~isfailure,
  hedit.String{end+1} = 'SUCCESS!'; drawnow;
elseif ~issuccess && isfailure,
  hedit.String{end+1} = sprintf('FAILURE. Could not create file in CacheDir %s:',cacheDir); drawnow;
  return;
else
  hedit.String{end+1} = 'FAILURE. ssh test failed.'; drawnow;
  return;
end

% test that we can run bjobs
hedit.String{end+1} = '** Testing that we can interact with the cluster...'; drawnow;
remotecmd = 'bjobs';
cmd = wrapCommandSSH(remotecmd,'host',host);
hedit.String{end+1} = cmd; drawnow;
[status,result] = apt.syscmd(cmd);
hedit.String{end+1} = result; drawnow;
if status ~= 0,
  hedit.String{end+1} = sprintf('Error running bjobs on %s',host); drawnow;
  return;
end
hedit.String{end+1} = 'SUCCESS!';
hedit.String{end+1} = '';
hedit.String{end+1} = 'All tests passed. JRC Backend should work for you.'; drawnow;

tfsucc = true;
