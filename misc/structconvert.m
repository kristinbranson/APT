function s = structconvert(s,fcn)
flds = fieldnames(s);
for f=flds(:)',f=f{1}; %#ok<FXSET>
  v = s.(f);
  if isstruct(v)
    s.(f) = structconvert(v,fcn);
  else
    s.(f) = fcn(v);
  end
end