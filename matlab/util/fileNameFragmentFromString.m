function result = fileNameFragmentFromString(str)
% Convert an arbitrary string into a fragment safe for use in a filename:
% delete all characters that are not letters, digits, spaces,
% underscores, or hyphens, then replace all spaces with underscores.
% Returns '' for empty input.
if isempty(str)
  result = '' ;
  return
end
keptChars = regexprep(str, '[^a-zA-Z0-9 _-]', '') ;
result = strrep(keptChars, ' ', '_') ;
end  % function
