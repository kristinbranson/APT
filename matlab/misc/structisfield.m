function tf = structisfield(s,fn)

if ischar(fn),
  fn = strsplit(fn,'.');
  tf = structisfield(s,fn);
  return;
end
if isempty(fn),
  tf = true;
  return;
end
if ~isstruct(s),
  tf = false;
  return;
end

if isfield(s,fn{1}),
  tf = structisfield(s.(fn{1}),fn(2:end));
else
  tf = false;
end
  