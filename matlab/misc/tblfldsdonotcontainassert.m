function tblfldsdonotcontainassert(t,flds)
flds = cellstr(flds);
assert(~any(ismember(flds(:),t.Properties.VariableNames(:))),...
  'Table fields contain unexpected entries.');