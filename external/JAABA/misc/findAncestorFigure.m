function fig=findAncestorFigure(handle)

atRoot=false;
foundIt=false;
fig=nan;
while ~foundIt && ~atRoot,
  type=get(handle,'type');
  if isequal(type,'figure')
    foundIt=true;
    fig=handle;
  elseif isequal(type,'root')
    atRoot=true;
  else
    handle=get(handle,'parent');
  end
end

end
