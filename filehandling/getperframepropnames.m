function props = getperframepropnames(trx)

nflies = length(trx);
props = fieldnames(trx);
ignoreprops = {'x','y','a','b','dx','dy'};
props = setdiff(props,ignoreprops);
if isfield(trx,'units'),
  props2 = fieldnames(trx(1).units);
  props = intersect(props2,props);
end
ignoreidx = [];
for i = 1:length(props),
  for j = 1:nflies,
    expectedlength = trx(j).nframes-1;
    if prod(size(trx(j).(props{i}))) < expectedlength,
      ignoreidx(end+1) = i;
      break;
    end
  end
end
props(ignoreidx) = [];
