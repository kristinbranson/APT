% [starts,ends] = get_interval_ends(x)
function [starts,ends] = get_interval_ends(x)

if size(x,1) > size(x,2),
  isrow = false;
else
  isrow = true;
end
x = [false;x(:);false];
% find starts and ends of sequences of 1s
starts = find((x(2:end)~=0) & (x(1:end-1)==0));
ends = find((x(2:end)==0) & (x(1:end-1)~=0));
if isrow,
  starts = starts';
  ends = ends';
end