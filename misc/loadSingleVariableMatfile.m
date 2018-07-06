function v = loadSingleVariableMatfile(fname)
vars = load(fname);
varnames = fieldnames(vars);
if ~isscalar(varnames)
  error('Expected a single variable in mat-file %s.\n',fname);
end
v = vars.(varnames{1});