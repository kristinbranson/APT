function val = structgetfield(s,fns)

if ischar(fns),
  fns = strsplit(fns,'.');
  val = structgetfield(s,fns);
  return;
end
val = [];
if isempty(fns) || ~isstruct(s),
  return;
end
for i = 1:numel(fns),
  fn = fns{i};
  if ~isfield(s,fn),
    return;
  end
  s = s.(fn);
end
val = s;