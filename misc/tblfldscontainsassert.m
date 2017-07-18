function tblfldscontainsassert(t,flds)
flds = cellstr(flds);
assert(all(ismember(flds(:),t.Properties.VariableNames(:))),...
  'Table fields do not contain expected entries.');