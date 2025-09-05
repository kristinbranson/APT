function result = wsl_fspollargs_from_native(native_fspollargs)
  % Given a cellstring suitable for passing to DLBackEndClass::batchPoll(), 
  % e.g. of the form {'exists' '/my/file' 'existsNE' '/my/file2'}, which uses
  % native paths, convert all the paths to WSL paths.
  % Note that all the 'paths' are apt.MetaPaths now.

  assert(iscellstr(native_fspollargs)) ;  %#ok<ISCLSTR>
  fspollargs_count = numel(native_fspollargs) ;
  assert(mod(fspollargs_count,2)==0) ;  % has to be even
  pair_count = fspollargs_count/2 ;
  result = native_fspollargs ;
  for pair_index = 1 : pair_count ,
    arg_index = 2*pair_index ;  % 2nd member of each pair is a path
    native_path = native_fspollargs{arg_index} ;
    wsl_path = native_path.asWsl() ;
    result{arg_index} = wsl_path ;
  end
end
