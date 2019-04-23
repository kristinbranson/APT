% bw = set_interval_ends(t0,t1,n)
function bw = set_interval_ends(t0,t1,n)

if ~exist('n','var')
  n = t1(end);
end

bw = false(1,n);
for i = 1:length(t0),
  bw(min(t0(i),n):min(t1(i),n)) = true;
end