function t = tblfldsmodify(t,fcn)
t.Properties.VariableNames = cellfun(fcn,t.Properties.VariableNames,'uni',0);