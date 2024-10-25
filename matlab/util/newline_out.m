function result = newline_out(lst)
% Concatenate the elements of a cellstring to yield a single string,
% putting newlines between adjacent elements.  (Name is by analogy with
% space_out().)

if isempty(lst)
  result = '';
  return
end
result = lst{1} ;
for i = 2:length(lst) ,
  result = horzcat(result, newline(), lst{i}) ;  %#ok<AGROW>
end

end
