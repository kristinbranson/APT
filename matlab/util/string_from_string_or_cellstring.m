function result = string_from_string_or_cellstring(thing)    
% Convert a char array, or string, or cell array of either of those,
% to a single char array.

if iscell(thing) ,
  if isempty(thing) ,
    result = '' ;
  else
    result = char(thing{1}) ;
    n = numel(thing) ;
    for j = 2 : n ,  % if n==1, this loop will do nothing
      subthing = thing{j} ;
      result = strcat(result, ',', char(subthing)) ;
    end
  end
else
  % Should be a char array or string
  result = char(thing) ;
end
