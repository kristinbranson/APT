function result = space_out(lst)
% Concatenate the elements of a cellstring to yield a single string,
% putting spaces between adjacent elements.

% if isempty(lst)
%   result = '';
%   return
% end
% result = lst{1} ;
% for i = 2:length(lst) ,
%   result = horzcat(result, ' ', lst{i}) ;  %#ok<AGROW>
% end

if isempty(lst)
  result = '' ;
else
  protoresult = sprintf('%s ',lst{:});
  result = protoresult(1:end-1) ;
end
