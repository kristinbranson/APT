% s = structappend(sbase,idx,snew)
function s = structarrayset(sbase,idx,snew)

if isempty(sbase),
  s(idx) = snew;
  return;
end
fnsnew = fieldnames(snew);
fnsbase = fieldnames(sbase);
fnsmissing = setdiff(fnsnew,fnsbase);
for i = 1:numel(fnsmissing),
  fn = fnsmissing{i};
  sbase(1).(fn) = [];
end
fnsmissing = setdiff(fnsbase,fnsnew);
for i = 1:numel(fnsmissing),
  fn = fnsmissing{i};
  snew(1).(fn) = [];
end
s = sbase;
s(idx) = orderfields(snew,s);
