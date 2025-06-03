function result = tidy_struct_from_yaml(x)
  % Recurses through a scalar struct that came of yaml, and tries to make it
  % nicer.  Currently, replaces cell arrays that cleanly convert to numeric
  % arrays with cell2mat() with the cell2mat() output.
  if isstruct(x)
    if ~isscalar(x) ,
      error('All struct elements must be scalar') ;
    end
    % If a (scalar) struct, recurse
    result = struct() ;
    field_names = fieldnames(x) ;
    for i = 1 : length(field_names) ,
      field_name = field_names{i} ;
      if strcmp(field_name, 'TransTypes')
        nop() ;
      end
      % If the field's value is a struct, recurse
      value = x.(field_name) ;
      result.(field_name) = yaml.tidy_struct_from_yaml(value) ;
    end
  elseif iscell(x) 
    try
      a = cell2mat(x) ;
      % Don't want to convert e.g. cell arrays of char arrays
      if isnumeric(a) || islogical(a) 
        result = a ;
      else
        result = x ;
      end
    catch
      result = x ;
    end
  else
    result =x ;
  end
end
