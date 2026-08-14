function s = structmvfield(s, oldfn, newfn)
% Move the value at the (possibly nested, dot-delimited) field path oldfn to
% the path newfn, then remove oldfn.  E.g. structmvfield(s, 'a.b', 'c.d')
% copies s.a.b to s.c.d and deletes s.a.b.

if ~structisfield(s, oldfn) ,
  return
end
value = getNestedField_(s, strsplit(oldfn, '.')) ;
s = setNestedField_(s, strsplit(newfn, '.'), value) ;
s = structrmfield(s, oldfn) ;
end  % function

function value = getNestedField_(s, parts)
% Get the value at the nested field path given by the cellstr parts.
if isscalar(parts) ,
  value = s.(parts{1}) ;
else
  value = getNestedField_(s.(parts{1}), parts(2:end)) ;
end
end  % function

function s = setNestedField_(s, parts, value)
% Set the value at the nested field path given by the cellstr parts,
% creating intermediate structs as needed.
if isscalar(parts) ,
  s.(parts{1}) = value ;
else
  if isfield(s, parts{1}) && isstruct(s.(parts{1})) ,
    subStruct = s.(parts{1}) ;
  else
    subStruct = struct() ;
  end
  s.(parts{1}) = setNestedField_(subStruct, parts(2:end), value) ;
end
end  % function
