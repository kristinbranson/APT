function result = wrapBatchCommandForAWSBackend(basecmd, backend)
  % Wrap a filesystem command for AWS, returns Linux/WSL-style command string.

  % Wrap for ssh'ing into an AWS instance
  cmd1 = backend.wrapCommandSSHAWS(basecmd) ;  % uses fields of awsec2 to set parameters for ssh command

  % Need to prepend a sleep to avoid problems
  precommand = 'sleep 5 && export AWS_PAGER=' ;
    % Change the sleep value at your peril!  I changed it to 3 and everything
    % seemed fine for a while, until it became a very hard-to-find bug!  
    % --ALT, 2024-09-12
  result = sprintf('%s && %s', precommand, cmd1) ;
end
