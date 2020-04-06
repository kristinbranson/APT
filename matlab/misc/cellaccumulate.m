function v = cellaccumulate(c,fcn)
assert(iscell(c) && ~isempty(c));
v = c{1};
for i=2:numel(c)
  v = fcn(v,c{i});
end

