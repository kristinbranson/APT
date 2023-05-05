function result = remove_repeated_slashes(str)
% Replace repeated /s with a single slash each.

result = regexprep(str, '/+', '/') ;

end
