function tblfldsassert(t,flds,varargin)

enforceorder = myparse(varargin,...
  'enforceorder',false);

if enforceorder
  assert(isequal(t.Properties.VariableNames(:),flds(:)),...
    'Table fields do not match specification.');
else
  assert(isequal(sort(t.Properties.VariableNames(:)),sort(flds(:))),...
    'Table fields do not match specification.');
end