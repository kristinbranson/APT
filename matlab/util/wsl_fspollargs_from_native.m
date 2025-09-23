function result = wsl_fspollargs_from_native(native_fspollargs)
  % Given a cell array suitable for passing to DLBackEndClass::batchPoll(), 
  % e.g. of the form {'exists' native_file_metapath 'existsNE' native_file2_metapath}, 
  % which uses native MetaPaths, convert all the paths to WSL MetaPaths.
  % Odd-indexed elements should be char arrays (commands), even-indexed should be MetaPaths (paths).

  assert(iscell(native_fspollargs), 'native_fspollargs must be a cell array');
  fspollargs_count = numel(native_fspollargs) ;
  assert(mod(fspollargs_count,2)==0) ;  % has to be even
  
  % Validate that odd-indexed elements are char arrays and even-indexed are MetaPaths
  for i = 1:fspollargs_count
    if mod(i, 2) == 1  % odd-indexed (commands)
      assert(ischar(native_fspollargs{i}), 'Odd-indexed elements must be char arrays (commands)');
    else  % even-indexed (paths)
      assert(isa(native_fspollargs{i}, 'apt.MetaPath'), 'Even-indexed elements must be apt.MetaPaths');
      assert(native_fspollargs{i}.locale == apt.PathLocale.native, 'All MetaPaths must have native locale');
    end
  end
  
  pair_count = fspollargs_count/2 ;
  result = native_fspollargs ;
  for pair_index = 1 : pair_count ,
    arg_index = 2*pair_index ;  % 2nd member of each pair is a path
    native_path = native_fspollargs{arg_index} ;
    wsl_path = native_path.asWsl() ;
    result{arg_index} = wsl_path ;
  end
end
