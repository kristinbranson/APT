function v = set_interval_ends(t0s,t1s,n,names,behavior)

if nargin < 3 || isempty(n),
  n = max(t1s);
else
  assert(all(t1s-1 <= n));
end

isnames= false;
if nargin >= 5,
  isnames= true;
  labelidx = strcmp(names,behavior);
else
  labelidx = true(1,numel(t0s));
end

assert(all(t0s<=t1s));
if isnames,
  v = nan(1,n);
else
  v = false(1,n);
end

for i = 1:numel(t0s),
  v(t0s(i):t1s(i)-1) = labelidx(i);
end