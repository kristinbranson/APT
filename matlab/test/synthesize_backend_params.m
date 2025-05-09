function result = synthesize_backend_params(backend)
  % environment_base_name = 'apt_20230427_tf211_pytorch113_ampere' ;
  environment_base_name = 'apt-20250505-tf215-pytorch21-hopper' ;
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
    sif_name = sprintf('%s.sif', environment_base_name) ;
    sif_path = fullfile('/groups/branson/bransonlab/apt/sif', sif_name) ;
    result = { ...
      'singularity_image_path', sif_path, ...
      'jrcgpuqueue','gpu_a100', ...
      'jrcnslots',4, ...
      'jrcAdditionalBsubArgs',jrcAdditionalBsubArgs } ;    
  elseif strcmp(backend, 'aws')
    generalAWSParams = {'awsInstanceID', 'i-0da079e9b4d2d66b9'} ;
    if strcmp(user_name, 'taylora') ,
      personalAWSParams = { ...
        'awsKeyName', 'alt_taylora-ws4', ...
        'awsPEM', '/home/taylora/.ssh/alt_taylora-ws4.pem' } ;
    else
      error('You need to customize %s.m to contain your AWS key name and PEM file location', mfilename()) ; 
    end
    result = horzcat(generalAWSParams, personalAWSParams) ;
  elseif strcmp(backend, 'docker')
    result = { 'dockerimgroot', 'bransonlabapt/apt_docker', 'dockerimgtag', environment_base_name } ;
  elseif strcmp(backend, 'conda')
    result = { 'condaEnv', environment_base_name } ;
  end
end  % function    
