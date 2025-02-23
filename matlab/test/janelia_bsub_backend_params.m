function result = janelia_bsub_backend_params()
  if strcmp(get_user_name(), 'taylora') ,
    jrcAdditionalBsubArgs = '-P scicompsoft' ;
  else
    jrcAdditionalBsubArgs = '' ;
  end
  result = ...
    {'jrcgpuqueue','gpu_a100', ...
     'jrcnslots',4, ...
     'jrcAdditionalBsubArgs',jrcAdditionalBsubArgs} ;
end  % function    
