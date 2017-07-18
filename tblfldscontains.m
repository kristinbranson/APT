function tf = tblfldscontains(tbl,fld)
if ischar(fld)
  tf = any(strcmp(fld,tbl.Properties.VariableNames));
else
  tf = ismember(fld,tbl.Properties.VariableNames);
end