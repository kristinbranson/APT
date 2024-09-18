function result = wrapFilesystemCommandForAWSBackend(basecmd, backend)
  % Wrap a filesystem command for AWS, returns Linux/WSL-style command string.
  
  % Wrap for ssh'ing into an AWS instance
  result = backend.awsec2.wrapCommandSSH(basecmd) ;
end
