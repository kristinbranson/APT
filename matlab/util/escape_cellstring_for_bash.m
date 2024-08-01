function result = escape_cellstring_for_bash(lst)
% Apply escape_string_for_bash() to all elements of a cellstring (a cell
% array of character row vectors).

result = cellfun(@escape_string_for_bash, lst, 'UniformOutput', false) ;
end
