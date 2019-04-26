% out = getstructarrayfield(s,field)
% s: array of structures
% field: name of a field of every s(i)
% out: concatenation of all s(i).(field)

function out = getstructarrayfield(s,field,varargin)

numericscalar = myparse(varargin,...
  'numericscalar',false);

if length(s) == 1,
  out = s.(field);
  return;
end

out = cell(size(s));
[out{:}] = deal(s.(field));
if numericscalar
  % in particular, protect against empty values [] in vector struct arrays
  cellfun(@(x)assert(isscalar(x)&&isnumeric(x),'Expected numeric scalar field values.'),out);
end
out = cell2mat(out);