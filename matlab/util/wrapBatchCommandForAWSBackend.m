function result = wrapBatchCommandForAWSBackend(basecmd, backend_or_awsec2)
  % Wrap a filesystem command for AWS, returns Linux/WSL-style command string.

  % Sort out the backend/ec2 nonsense
  if isa(backend_or_awsec2, 'DLBackEndClass') ,
    backend = backend_or_awsec2 ;
    ec2 = backend.awsec2 ;
  elseif isa(backend_or_awsec2, 'AWSec2') ,
    ec2 = backend_or_awsec2 ;
  else
    error('The second argument must be an instance of the class DLBackEndClass or the class AWSec2') ;
  end

  % Wrap for ssh'ing into an AWS instance
  cmd1 = ec2.wrapCommandSSH(basecmd) ;  % uses fields of ec2 to set parameters for ssh command

  % Need to prepend a sleep to avoid problems
  precommand = 'sleep 5 && export AWS_PAGER=' ;
    % Change the sleep value at your peril!  I changed it to 3 and everything
    % seemed fine for a while, until it became a very hard-to-find bug!  
    % --ALT, 2024-09-12
  result = sprintf('%s && %s', precommand, cmd1) ;
end
