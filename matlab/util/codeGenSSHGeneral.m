function codestr = codeGenSSHGeneral(remotecmd,varargin)
  % Currently this assumes a JRC backend due to oncluster special case      
  [host,bg,prefix,sshoptions,timeout] = ...
    myparse(varargin,...
            'host',DLBackEndClass.jrchost,... % 'logfile','/dev/null',...
            'bg',false,... % AL 20201022 see note below
            'prefix',DLBackEndClass.jrcprefix,...
            'sshoptions','-oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR',...
            'timeout',[]);
  
  if ~isempty(prefix),
    remotecmd = [prefix,'; ',remotecmd];
  end
  if ~isempty(timeout),
    sshoptions1 = ['-o "ConnectTimeout ',num2str(timeout),'"'];
    if ~ischar(sshoptions) || isempty(sshoptions),
      sshoptions = sshoptions1;
    else
      sshoptions = [sshoptions,' ',sshoptions1];
    end
  end
      
  if ~ischar(sshoptions) || isempty(sshoptions),
    sshcmd = 'ssh';
  else
    sshcmd = ['ssh ',sshoptions];
  end
        
  if bg
    % AL 20201022 not sure why this codepath was nec. Now it is causing
    % problems with LSF/job scheduling. The </dev/null & business
    % confuses LSF and the account/runtime limit doesn't get set. So
    % for now this is a nonproduction codepath.
    codestr = sprintf('%s %s ''%s </dev/null &''',sshcmd,host,remotecmd);
  else
    tfOnCluster = ~isempty(getenv('LSB_DJOB_NUMPROC'));
    if tfOnCluster
      codestr = remotecmd;
    else
      codestr = sprintf('%s %s ''%s''',sshcmd,host,remotecmd);
    end
  end
end

