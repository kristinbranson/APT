function result = norm_string_if_empty(str)
  % If str is empty, make sure it is the canonical empty (old-school) string,
  % ''.  ('' is 0x0, which strcmp() cares about.)  If str is nonempty, return it
  % unchanged.
  if isempty(str), 
    result = '' ;
  else
    result = str ;
  end
end
