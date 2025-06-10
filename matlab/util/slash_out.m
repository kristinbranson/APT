function result = slash_out(lst)
% Concatenate the elements of a cellstring to yield a single string,
% putting slashes between adjacent elements.

if isempty(lst)
  result = '' ;
else
  protoresult = sprintf('%s/',lst{:});
  result = protoresult(1:end-1) ;
end
