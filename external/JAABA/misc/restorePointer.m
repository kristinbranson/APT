function restorePointer(handle,oldPointer)

drawnow('update');  % make sure everything is updated before we restore the pointer
fig=findAncestorFigure(handle);
currentPointer=get(fig,'pointer');
if ~strcmpi(oldPointer,currentPointer)
  set(fig,'pointer',oldPointer);
end
drawnow('update');

end
