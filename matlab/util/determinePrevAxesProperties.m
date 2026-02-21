function axes_curr = determinePrevAxesProperties(paModeInfo, axesCurrProps)
  % Returns a struct containing several properties of the "prev" axes,
  % determined partly from paModeInfo and partly from looking at the axes
  % graphics handle.  This method does not mutate obj.
  % axesCurrProps is a struct with fields XDir, YDir, XLim, YLim.
  xdir = axesCurrProps.XDir;
  ydir = axesCurrProps.YDir;
  if ~isfield(paModeInfo,'xlim'),
    xlim = axesCurrProps.XLim;
    ylim = axesCurrProps.YLim;
  else
    xlim = paModeInfo.xlim;
    ylim = paModeInfo.ylim;
  end
  axes_curr = struct('XLim',xlim,'YLim',ylim,...
                     'XDir',xdir','YDir',ydir,...
                     'CameraViewAngleMode','auto');
end  % function

