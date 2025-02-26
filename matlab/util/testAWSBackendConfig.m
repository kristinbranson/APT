function [tfsucc,hedit] = testAWSBackendConfig(backend)  %#ok<INUSD> 

tfsucc = false;
[~,hedit] = createFigTestBackendConfig('Test AWS Backend');
hedit.String = {sprintf('%s: Testing AWS backend...',datestr(now))}; 
drawnow;

% test that ssh exists
hedit.String{end+1} = sprintf('** Testing that ssh is available...'); drawnow;
hedit.String{end+1} = ''; drawnow;
if ispc,
  isssh = exist(APT.WINSSHCMD,'file') && exist(APT.WINSCPCMD,'file');
  if isssh,
    hedit.String{end+1} = sprintf('Found ssh at %s',APT.WINSSHCMD); 
    drawnow;
  else
    hedit.String{end+1} = sprintf('FAILURE. Did not find ssh in the expected location: %s.',APT.WINSSHCMD); 
    drawnow;
    return;
  end
else
  cmd = 'which ssh';
  hedit.String{end+1} = cmd; drawnow;
  [status,result] = apt.syscmd(cmd);
  hedit.String{end+1} = result; drawnow;
  if status ~= 0,
    hedit.String{end+1} = 'FAILURE. Did not find ssh.'; drawnow;
    return;
  end
end

if ispc(),
  hedit.String{end+1} = sprintf('\n** Testing that certUtil is installed...\n'); drawnow;
  cmd = 'where certUtil';
  hedit.String{end+1} = cmd; drawnow;
  [status,result] = apt.syscmd(cmd);
  hedit.String{end+1} = result; drawnow;
  if status ~= 0,
    hedit.String{end+1} = 'FAILURE. Did not find certUtil.'; drawnow;
    return;
  end
end

% test that AWS CLI is installed
hedit.String{end+1} = sprintf('\n** Testing that AWS CLI is installed...\n'); drawnow;
cmd = 'aws ec2 describe-regions --output table';
hedit.String{end+1} = cmd; drawnow;
[status,result] = AWSec2.syscmd(cmd);
tfsucc = (status==0) ;      
hedit.String{end+1} = result; drawnow;
if ~tfsucc % status ~= 0,
  hedit.String{end+1} = 'FAILURE. Error using the AWS CLI.'; drawnow;
  return
end

% test that apt_dl security group has been created
hedit.String{end+1} = sprintf('\n** Testing that apt_dl security group has been created...\n'); drawnow;
cmd = 'aws ec2 describe-security-groups';
hedit.String{end+1} = cmd; drawnow;
[status,result] = AWSec2.syscmd(cmd,'isjsonout',true);
tfsucc = (status==0) ;
if status == 0,
  try
    result = jsondecode(result);
    if ismember('apt_dl',{result.SecurityGroups.GroupName}),
      hedit.String{end+1} = 'Found apt_dl security group.'; drawnow;
    else
      status = 1;
    end
  catch
    status = 1;
  end
  if status == 1,
    hedit.String{end+1} = 'FAILURE. Could not find the apt_dl security group.'; drawnow;
  end
else
  hedit.String{end+1} = result; drawnow;
  hedit.String{end+1} = 'FAILURE. Error checking for apt_dl security group.'; drawnow;
  return
end

% to do, could test launching an instance, or at least dry run

%       m = regexp(result,' (\d+) received, (\d+)% packet loss','tokens','once');
%       if isempty(m),
%         hedit.String{end+1} = 'FAILURE. Could not parse ping output.'; drawnow;
%         return;
%       end
%       if str2double(m{1}) == 0,
%         hedit.String{end+1} = sprintf('FAILURE. Could not ping %s:\n',host); drawnow;
%         return;
%       end
%       hedit.String{end+1} = 'SUCCESS!'; drawnow;
%       
%       % test that we can connect to jrc host and access CacheDir on it
%      
%       hedit.String{end+1} = ''; drawnow;
%       hedit.String{end+1} = sprintf('** Testing that we can do passwordless ssh to %s...',host); drawnow;
%       touchfile = fullfile(cacheDir,sprintf('testBsub_test_%s.txt',datestr(now,'yyyymmddTHHMMSS.FFF')));
%       
%       remotecmd = sprintf('touch %s; if [ -e %s ]; then rm -f %s && echo "SUCCESS"; else echo "FAILURE"; fi;',touchfile,touchfile,touchfile);
%       cmd1 = wrapCommandSSH(remotecmd,'host',host);
%       cmd = sprintf('timeout 20 %s',cmd1);
%       hedit.String{end+1} = cmd; drawnow;
%       [status,result] = apt.syscmd(cmd);
%       hedit.String{end+1} = result; drawnow;
%       if status ~= 0,
%         hedit.String{end+1} = sprintf('ssh command timed out. This could be because passwordless ssh to %s has not been set up. Please see APT wiki for more details.',host); drawnow;
%         return;
%       end
%       issuccess = contains(result,'SUCCESS');
%       isfailure = contains(result,'FAILURE');
%       if issuccess && ~isfailure,
%         hedit.String{end+1} = 'SUCCESS!'; drawnow;
%       elseif ~issuccess && isfailure,
%         hedit.String{end+1} = sprintf('FAILURE. Could not create file in CacheDir %s:',cacheDir); drawnow;
%         return;
%       else
%         hedit.String{end+1} = 'FAILURE. ssh test failed.'; drawnow;
%         return;
%       end
%       
%       % test that we can run bjobs
%       hedit.String{end+1} = '** Testing that we can interact with the cluster...'; drawnow;
%       remotecmd = 'bjobs';
%       cmd = wrapCommandSSH(remotecmd,'host',host);
%       hedit.String{end+1} = cmd; drawnow;
%       [status,result] = apt.syscmd(cmd);
%       hedit.String{end+1} = result; drawnow;
%       if status ~= 0,
%         hedit.String{end+1} = sprintf('Error running bjobs on %s',host); drawnow;
%         return;
%       end
hedit.String{end+1} = 'SUCCESS!'; 
hedit.String{end+1} = ''; 
hedit.String{end+1} = 'All tests passed. AWS Backend should work for you.'; drawnow;

tfsucc = true;      
end
