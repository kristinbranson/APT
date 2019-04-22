% [spath,sfilename] = split_path_and_filename(s)
function [spath,sfilename] = split_path_and_filename(s)

i = find(s(1:end-1)=='/',1,'last');
if isempty(i),
  % windows?
  i = find(s(1:end-1)=='\',1,'last');
  if isempty(i),
    spath = '';
    sfilename = s;
    return;
  end
end
spath = s(1:i);
sfilename = s(i+1:end);
