function result = struct_from_key_value_list(key_value_list)
  % key_value_list should be a cell array of keys and values.  
  % E.g. { 'foo', 1, 'bar', 2 }
  % results is a scalar struct with the keys as the field names, and each field
  % containing the correspodning value.
  % E.g. struct('foo', {1}, 'bar', {2})
  key_from_pair_index = key_value_list(1:2:end) ;
  value_from_pair_index = key_value_list(2:2:end) ;
  key_count = numel(key_from_pair_index) ;
  value_count = numel(value_from_pair_index) ;
  if key_count ~= value_count ,
    error('Must have same number of keys as values') ;
  end
  celled_value_from_pair_index = cellfun(@(value)({value}), value_from_pair_index, 'UniformOutput', false) ;
    % A cell array, with each cell containing a singleton cell array, each
    % singleton cell array holding a single value.
  pair_from_pair_index = cellfun(@(key, value)({key, value}), key_from_pair_index, celled_value_from_pair_index, 'UniformOutput', false) ;
  argument_list = horzcat(pair_from_pair_index{:}) ;
  result = struct(argument_list{:}) ;
end
