function result = isTableColumn(tbl, queryColName)

colNames = tbl.Properties.VariableNames ;
result = ismember(queryColName, colNames) ;

end
