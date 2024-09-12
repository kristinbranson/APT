function cmdout = wrapCommandSSH(remotecmd,varargin)

[host,prefix,sshoptions,timeout,extraprefix] = ...
  myparse(varargin,...
          'host',DLBackEndClass.jrchost,...
          'prefix',DLBackEndClass.jrcprefix,...
          'sshoptions','-oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR -t',...
          'timeout',[],...
          'extraprefix','');

if ~isempty(extraprefix),
  prefix = sprintf('%s; %s', prefix, extraprefix) ;
end

if ~isempty(prefix),
  remotecmd = sprintf('%s; %s', prefix, remotecmd) ;  
end

quotedremotecmd = escape_string_for_bash(remotecmd);

if ~isempty(timeout),
  sshoptions1 = sprintf('-o "ConnectTimeout %s"', num2str(timeout)) ;
  if ~ischar(sshoptions) || isempty(sshoptions),
    sshoptions = sshoptions1;
  else
    sshoptions = sprintf('%s %s', sshoptions, sshoptions1) ;    
  end
end
if ~ischar(sshoptions) || isempty(sshoptions),
  sshcmd = 'ssh';
else
  sshcmd = sprintf('ssh %s',sshoptions) ;
end

cmdout = sprintf('%s %s %s',sshcmd,host,quotedremotecmd);
