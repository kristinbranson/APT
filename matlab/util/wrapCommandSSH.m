function result = wrapCommandSSH(baseCommand, varargin)
% Wrap a Linux/WSL-style command for execution on a remote host via
% ssh.  baseCommand should be a ShellCommand with remote locale.

% Validate input baseCommand
assert(isa(baseCommand, 'apt.ShellCommand'), 'baseCommand must be an apt.ShellCommand object');
assert(baseCommand.tfDoesMatchLocale(apt.PathLocale.remote), 'baseCommand must have remote locale');

% Deal with keyword arguments
nilCommand = apt.ShellCommand({}, apt.PathLocale.remote, apt.Platform.posix);
[host,prefix,sshoptions,addlsshoptions,timeout,extraprefix,username,identityWslMetaPathOrEmpty] = ...
  myparse(varargin,...
          'host','',...
          'prefix',nilCommand,...
          'sshoptions','',...
          'addlsshoptions','',...
          'timeout',[],...
          'extraprefix',nilCommand, ...
          'username','', ...
          'identity',[]);

% Sort out the prefixes, merging them all into prefixCommand
if ~extraprefix.isNull()
  if prefix.isNull()
    fullPrefix = extraprefix ;
  else
    fullPrefix = prefix.cat(';', extraprefix);
  end
else
  fullPrefix = prefix;
end

% Append the fullPrefix, if present, to remoteCommand
if fullPrefix.isNull()
  prefixedBaseCommand = baseCommand ;
else
  prefixedBaseCommand = fullPrefix.cat(';', baseCommand) ;
end

% Sort out the sshoptions and the addlsshoptions
baseSshOptions = apt.ShellCommand({'-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null', '-o', 'LogLevel=ERROR', '-t'}, ...
                                  apt.PathLocale.wsl, ...
                                  apt.Platform.posix);
if isempty(sshoptions) 
  if isempty(addlsshoptions) 
    % Both empty, use the base options
    sshOptionsCommand1 = baseSshOptions ;
  else
    % sshoptions empty, addlsshoptions nonempty => add addlsshoptions to base options
    sshOptionsCommand1 = baseSshOptions.cat(addlsshoptions) ;
  end
else
  if isempty(addlsshoptions) 
    % sshoptions nonempty, addlsshoptions empty => use sshoptions, overriding base_sshoptions
    sshOptionsCommand1 = apt.ShellCommand({sshoptions}, apt.PathLocale.wsl, apt.Platform.posix) ;
  else
    % sshoptions nonempty, addlsshoptions nonempty => use sshoptions, overriding
    % base_sshoptions, thus ignoring addlsshoptions.  But give a warning about
    % this unusual (and probably unintentional) usage.
    warning('sshoptions and addlsshoptions both provided: ignoring addlsshoptions') ;  
    sshOptionsCommand1 = apt.ShellCommand({sshoptions}, apt.PathLocale.wsl, apt.Platform.posix) ;
  end
end  

% Append the timeout duration, if provided, to the ssh options
if isempty(timeout)
  sshOptionsCommand2 = sshOptionsCommand1 ;
else
  sshOptionsCommand2 = sshOptionsCommand1.append('-o', sprintf('ConnectTimeout=%d', round(timeout))) ;
end

% Append the identity file, if provided, to the ssh options
if isempty(identityWslMetaPathOrEmpty)
  sshOptionsCommand3 = sshOptionsCommand2 ;
else
  identityWslMetaPath = identityWslMetaPathOrEmpty;
  assert(isa(identityWslMetaPath, 'apt.MetaPath'), 'identity must be an apt.MetaPath');
  assert(identityWslMetaPath.locale == apt.PathLocale.wsl, 'identity MetaPath must have WSL locale');
  assert(identityWslMetaPath.role == apt.FileRole.local, 'identity MetaPath must have local FileRole');
  dashEyeArg = apt.ShellCommand({'-i', identityWslMetaPath}, apt.PathLocale.wsl, apt.Platform.posix);
  sshOptionsCommand3 = dashEyeArg.cat(sshOptionsCommand2);
end

% Append the username, if provided, to the hostname
if ~isempty(username) 
  userAtHostString = sprintf('%s@%s', username, host) ;  
else
  userAtHostString = host ;
end

% Generate the final command line
command0 = apt.ShellCommand({'ssh'}, apt.PathLocale.wsl, apt.Platform.posix);
command1 = command0.cat(sshOptionsCommand3);
command2 = command1.append(userAtHostString);
command3 = command2.append(prefixedBaseCommand);  % Important to append, not cat

result = command3 ;

end  % function
