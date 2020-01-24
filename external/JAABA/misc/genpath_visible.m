% p = genpath_visible(d)
% save as genpath, but if any of the path is hidden, the directory is not
% added
function p = genpath_visible(d)

hiddenchar = '.';

p = '';
a = genpath(d);
if isempty(a),
  return;
end

% find the first character of each path
i0s = [0,find(a == pathsep)]+1;
if i0s(end) <= length(a),
  i0s(end+1) = length(a)+1;
end
n = length(i0s)-1;
off = length(d) + 1;

for i = 1:n,
  i0 = i0s(i);
  i1 = i0s(i+1)-2;
  if i0+off > i1,
    continue;
  end
  ishidden = a(i0+off) == hiddenchar || ...
    ~isempty(strfind(a(i0+off:i1),[filesep,hiddenchar]));
  if ~ishidden,
    p = [p,a(i0:i1),pathsep]; %#ok<AGROW>
    %fprintf('Added: %s\n',a(i0:i1));
    %else
    %fprintf('Hidden: %s\n',a(i0:i1));
  end
end
