function lObj = getLabelerObj

h = findall(0,'type','figure','-regexp','name','APT');
if isempty(h)
  error('Cannot find APT figure window.');
end

if numel(h)>1
  warningNoTrace('Found more than one figure window with name matching ''APT''.');
end

lObj = [];
for i=1:numel(h)
  gd = guidata(h(i));
  if isfield(gd,'labelerObj') && isa(gd.labelerObj,'Labeler')
    lObj = gd.labelerObj;
    break;
  end
end

if isempty(lObj)
  warningNoTrace('Could not find Labeler object.');
end