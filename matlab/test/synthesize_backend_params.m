function result = synthesize_backend_params(backend)
  % Synthesize a name-value list of backend parameters for use in the test suite.
  % The image/environment name is taken from the DLBackEndClass defaults in all
  % cases, so that tests run against whatever image/environment the class considers
  % current, and so that a stale image/environment override baked into a loaded
  % project gets replaced by the class default at test time.
  user_name = get_user_name() ;
  % We could just result a list with params for *all* backends, but then we
  % would have to error if user has not customized this function to add *their*
  % AWS info, even if they're not using the AWS backend.
  if strcmp(backend, 'bsub')
    if strcmp(user_name, 'taylora') ,
      jrcAdditionalBsubArgs = '-P scicompsoft' ;
    else
      jrcAdditionalBsubArgs = '' ;
    end
    result = { ...
      'singularity_image_path', DLBackEndClass.DEFAULT_SINGULARITY_IMAGE_PATH, ...
      'jrcgpuqueue','gpu_a100', ...
      'jrcnslots',4, ...
      'jrcAdditionalBsubArgs',jrcAdditionalBsubArgs } ;
  elseif strcmp(backend, 'aws')
    generalAWSParams = {'awsInstanceID', 'i-0c893bbd7b6cf1853'} ;
    if strcmp(user_name, 'taylora') ,
      personalAWSParams = { ...
        'awsKeyName', 'alt_taylora-ws4', ...
        'awsPEM', '/home/taylora/.ssh/alt_taylora-ws4.pem' } ;
    else
      error('You need to customize %s.m to contain your AWS key name and PEM file location', mfilename()) ;
    end
    result = horzcat(generalAWSParams, personalAWSParams) ;
  elseif strcmp(backend, 'docker')
    result = { ...
      'dockerimgroot', DLBackEndClass.defaultDockerImgRoot, ...
      'dockerimgtag', DLBackEndClass.defaultDockerImgTag } ;
  elseif strcmp(backend, 'conda')
    result = { 'condaEnv', DLBackEndClass.default_conda_env } ;
  else
    error('APT:invalidValue', 'Unknown backend type "%s"', backend) ;
  end
end  % function
