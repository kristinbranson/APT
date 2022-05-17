% v = isSubField(s,fns)
% whether all s.(fns{1}).(fns{2)}. ... fields exist

function v = isSubField(s,fns)

if numel(fns) == 1,
  v = isfield(s,fns{1});
else
  v = isfield(s,fns{1}) && isSubField(s.(fns{1}),fns(2:end));
end