function result = strcatg(varargin)
% Like strcat(), but doesn't remove whitespace for char array arguments.
% The "g" is for "general".  If all args are char arrays will use horzcat() to
% concatenate them.  If any is a string, will convert all args to strings,
% then use strcat() to concatenate them.

class_name_from_argument_index = cellfun(@class, varargin, 'UniformOutput', false) ;
if any(strcmp('string', class_name_from_argument_index)) ,
  arguments_as_strings = cellfun(@string, varargin, 'UniformOutput', false) ;  % 1xn arrary of strings
  result = strcat(arguments_as_strings{:}) ;
else
  % Assume all char arrays
  result = horzcat(varargin{:}) ;
end
