function result = is_valid_function_name(name)
  % Determine if name (a string) is the name of a currently-valid matlab function.
  % From: https://www.mathworks.com/matlabcentral/answers/120402-how-to-know-if-a-string-is-a-valid-matlab-function
  result = regexp(name, '^[A-Za-z]\w*$') && ismember(exist(name), [2 3 5 6]) ;  %#ok<EXIST>
end
