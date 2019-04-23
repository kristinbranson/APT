% [path,base] = filenamesplit(name)
function [path,base] = filenamesplit(name)

while 1,
  if name(end) == filesep,
    name = name(1:end-1);
  else
    break;
  end
end

k = strfind(name,filesep);
if isempty(k),
  path = '';
  base = name;
  return;
end

path = name(1:k(end));
base = name(k(end)+1:end);