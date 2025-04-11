function cmdout = wrapCommandSSH(remotecmd,varargin)
% Wrap a Linux/WSL-style command string for execution on a remote host via
% ssh.  Output is a Linux/WSL-style string, regardless of ispc() return value.

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

% Sort out the prefixes, merging them all into prefixes
if isempty(extraprefix),
  if isempty(prefix) ,
    prefixes = '' ;
  else
    prefixes = prefix ;
  end
else
  if isempty(prefix) ,
    prefixes = extraprefix ;
  else
    prefixes = sprintf('%s ; %s', prefix, extraprefix) ;
  end
end

% Append the prefixes, if present, to remotecmd
if isempty(prefixes),
  prefixed_remotecmd = remotecmd ;
else
  prefixed_remotecmd = sprintf('%s ; %s', prefixes, remotecmd) ;  
end

% quote the prefixes command
quoted_prefixed_remotecmd = escape_string_for_bash(prefixed_remotecmd);

% Sort out the sshoptions and the addlsshoptions
if isempty(sshoptions) ,
  base_sshoptions = '-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -t' ;
  if isempty(addlsshoptions) ,
    % Both empty, use the base options
    sshoptions2 = base_sshoptions ;
  else
    % sshoptions empty, addlsshoptions nonempty => add addlsshoptions to base
    % options
    sshoptions2 = sprintf('%s %s', base_sshoptions, addlsshoptions) ;
  end
else
  if isempty(addlsshoptions) ,
    % sshoptions nonempty, addlsshoptions empty => use sshoptions, overriding
    % base_sshoptions
    sshoptions2 = sshoptions ;
  else
    % sshoptions nonempty, addlsshoptions nonempty => use sshoptions, overriding
    % base_sshoptions, thus ignoring addlsshoptions.  But give a warning about
    % this unusual (and probably unintentional) usage.
    warning('sshoptions and addlsshoptions both provided: ignoring addlsshoptions') ;  
    sshoptions2 = sshoptions ;
  end
end  

% Append the timeout duration, if provided, to the ssh options
if isempty(timeout),
  sshoptions3 = sshoptions2 ;
else
  sshoptions3 = sprintf('%s -o ConnectTimeout=%d', sshoptions2, round(timeout)) ;
end

% Append the idenitity file, if provided, to the ssh options
if isempty(identity),
  sshoptions4 = sshoptions3 ;
else
  quoted_identity = escape_string_for_bash(identity) ;
  sshoptions4 = sprintf('-i %s %s', quoted_identity, sshoptions3) ;
end

% Append the username, if provided, to the hostname
if ~isempty(username) ,
  user_at_host_string = sprintf('%s@%s', username, host) ;  
else
  user_at_host_string = host ;
end

% Generate the final command line
cmdout = sprintf('ssh %s %s %s', sshoptions4, user_at_host_string, quoted_prefixed_remotecmd) ;
