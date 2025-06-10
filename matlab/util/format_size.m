function result = format_size(sz)
  % Output a string (char row vector) representing the array dimension vector
  % sz.  E.g. [ 1 2 3 ] => '1x2x3'
  if isempty(sz) ,
    result = '<zero-dimensional>' ;
  elseif isscalar(sz) ,
    result = sprintf('%d', sz) ;
  else
    head = sz(1) ;
    rest = sz(2:end) ;
    prefix = sprintf('%dx', head) ;
    result = horzcat(prefix, format_size(rest)) ;
  end    
end
