function result = strrep_multiple(str, patterns, replacements)
% In the string str, replace each occurence of a string in patterns with the
% corresponding element of replacements. Replacements are done sequentially,
% so if some pattern is a substring of another pattern, the result may not be
% what you expect.
result = str ;
n = numel(patterns) ;
for i = 1 : n ,
  pattern = patterns{i} ;
  replacement = replacements{i} ;
  result = strrep(result, pattern, replacement) ;
end
