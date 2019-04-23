function adjustButtonColorsIfMac(buttons)
% This adjust the color of non-label buttons for the platform.  On Windows
% and Linux, this does nothing.  On Mac, it gives the buttons the default
% colors.

if ismac()
  defaultBackgroundColor = get(0,'defaultUicontrolBackgroundColor');
  set(buttons,'ForegroundColor','k', ...
              'BackgroundColor',defaultBackgroundColor, ...
              'Cdata',[]);  
end
                
end
