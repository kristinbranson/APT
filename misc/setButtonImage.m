function setButtonImage(hObj)
  
bgColor = get(hObj,'BackgroundColor');
buttonPos = get(hObj,'Position');
buttonSz = round(buttonPos([4 3]));
buttonImg = shiftdim(repmat(bgColor,[buttonSz(2) 1 buttonSz(1)]),2);
if ismac
  if strcmpi(get(hObj,'Units'),'pixels'),
    set(hObj,'CData',buttonImg);
  else
    set(hObj,'BackgroundColor',[1 1 1]);
    set(hObj,'ForegroundColor',[0 0 0]);
  end
end
