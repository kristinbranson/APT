function result = flatten_row_cell_array(a)
  % Takes a row cell array of row cell arrays (to whatever depth)
  % and returns a flat row cell array.
  if iscell(a)
    flattened = cellfun(@flatten_row_cell_array, a, 'UniformOutput', false) ;    
    result = horzcat(flattened{:}) ;
  else
    % Want non-cell things to be wrapped up in cells.
    % This is important so that e.g. 
    %   flatten_row_cell_array({'a', 'b'})
    % doesn't return 'ab'.
    result = {a} ;
  end
end
