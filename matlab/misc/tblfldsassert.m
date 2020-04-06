function tblfldsassert(t,flds,varargin)

enforceorder = myparse(varargin,...
  'enforceorder',false);

varNames = t.Properties.VariableNames;
varNames = varNames(:); % AL20180213: columnizing must be separate from indexing in 2014b
if enforceorder
  assert(isequal(varNames,flds(:)),'Table fields do not match specification.');
else
  assert(isequal(sort(varNames),sort(flds(:))),...
    'Table fields do not match specification.');
end