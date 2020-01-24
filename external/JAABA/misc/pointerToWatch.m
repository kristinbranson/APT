function oldPointer=pointerToWatch(handle)

fig=findAncestorFigure(handle);
oldPointer=get(fig,'pointer');
if ~strcmpi(oldPointer,'watch')
  set(fig,'pointer','watch');
end
drawnow('update');

end
