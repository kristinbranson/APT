function result = wrapCommandSSH(baseCommand, varargin)
% Wrap a Linux/WSL-style command for execution on a remote host via
% ssh.  baseCommand should be a ShellCommand with WSL locale.

% Validate input baseCommand
assert(isa(baseCommand, 'apt.ShellCommand'), 'baseCommand must be an apt.ShellCommand object');
assert(baseCommand.tfDoesMatchLocale(apt.PathLocale.wsl), 'baseCommand must have WSL locale');

% Deal with keyword arguments
[host,prefix,sshoptions,addlsshoptions,timeout,extraprefix,username,identity] = ...
  myparse(varargin,...
          'host','',...
          'prefix','',...
          'sshoptions','',...
          'addlsshoptions','',...
          'timeout',[],...
          'extraprefix','', ...
          'username','', ...
          'identity','');

% Sort out the prefixes, merging them all into prefixCommand
prefix0 = apt.ShellCommand({}, apt.PathLocale.remote, apt.Platform.posix);
if ~isempty(prefix)
  prefix1 = prefix0.append(prefix);
else
  prefix1 = prefix0;
end
if ~isempty(extraprefix)
  if prefix1.isempty()
    prefix2 = apt.ShellCommand({extraprefix}, apt.PathLocale.remote, apt.Platform.posix);
  else
    prefix2 = apt.ShellCommand.cat(prefix1, ';', extraprefix);
  end
else
  prefix2 = prefix1;
end

% Append the prefixes, if present, to remoteCommand
if prefix2.isempty()
  prefixedBaseCommand = apt.ShellCommand({baseCommand}, apt.PathLocale.remote, apt.Platform.posix);
else
  prefixedBaseCommand = apt.ShellCommand.cat(prefix2, ';', baseCommand);
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
    sshOptionsCommand1 = apt.ShellCommand.cat(baseSshOptions, addlsshoptions) ;
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
if isempty(identity)
  sshOptionsCommand3 = sshOptionsCommand2 ;
else
  identityPath = apt.MetaPath(identity, 'native', 'universal');
  sshOptionsCommand3 = apt.ShellCommand({'-i', identityPath}, apt.PathLocale.wsl, apt.Platform.posix).cat(sshOptionsCommand2) ;
end

% Append the username, if provided, to the hostname
if ~isempty(username) 
  userAtHostString = sprintf('%s@%s', username, host) ;  
else
  userAtHostString = host ;
end

% Generate the final command line
command0 = apt.ShellCommand({'ssh'}, apt.PathLocale.wsl, apt.Platform.posix);
command1 = apt.ShellCommand.cat(command0, sshOptionsCommand3);
command2 = command1.append(userAtHostString);
command3 = apt.ShellCommand.cat(command2, prefixedBaseCommand);

result = command3 ;

end  % function
