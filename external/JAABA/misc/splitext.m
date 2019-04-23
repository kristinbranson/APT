% [base,ext] = splitext(name)
function [base,ext] = splitext(name)

k = strfind(name,'.');
if isempty(k),
  base = name;
  ext = '';
  return;
end

k = k(end);
j = strfind(name,'/');
if ~isempty(j) && j(end) > k,
  base = name;
  ext = '';
  return;
end

base = name(1:k-1);
ext = name(k:end);