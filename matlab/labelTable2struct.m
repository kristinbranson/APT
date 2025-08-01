function s = labelTable2struct(tbl)

cols = tbl.Properties.VariableNames;
s = struct;
for i = 1:numel(cols),
  col = cols{i};
  s.(col) = tbl.(col);
end

