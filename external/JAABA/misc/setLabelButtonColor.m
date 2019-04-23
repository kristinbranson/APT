function setLabelButtonColor(button,clr)
% Sets the button color to clr.  What exactly this means depends on the 
% platform.  On Windows and Linux, the background color is set to a lower-intesity
% version of the given color, and the text color is set to white.  On
% Mac, the text color is set to clr, and the button background is set to
% the default color.  button is a handle to a uicontrol.

if ismac()
  defaultBackgroundColor = get(0,'defaultUicontrolBackgroundColor');
  set(button,'ForegroundColor',clr, ...
             'BackgroundColor',defaultBackgroundColor, ...
             'CData',[]);  
elseif ispc,
  set(button,'ForegroundColor',[.99,.99,.99], ...
    'BackgroundColor', clr, ...
    'CData',[]);
  
else
  set(button,'ForegroundColor','w', ...
             'BackgroundColor', clr, ...
             'CData',[]);
end

end
