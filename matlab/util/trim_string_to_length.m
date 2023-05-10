function result = trim_string_to_length(str, max_length)
% Trim this string to a maximum length of n, putting an ellipsis at the
% beginning to indicate the truncation.

str_length = length(str) ;
if str_length > max_length ,
  if max_length<3 ,
    error('The max_length must be at least three, to accomodate the ellipses') ;
  end
  i_first = str_length - max_length + 4 ;  % guaranteed to be positive
  i_last = str_length ;
  result = horzcat('...', str(i_first:i_last)) ;
else
  result = str ;
end

end