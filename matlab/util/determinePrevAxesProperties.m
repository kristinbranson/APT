function axes_curr = determinePrevAxesProperties(cache, axesCurrProps)
  % Returns a struct containing several properties of the "prev" axes,
  % determined partly from cache and partly from looking at the axes
  % graphics handle.  This method does not mutate obj.
  % cache is a struct with cache fields from prevAxesModeTargetCache_.
  % axesCurrProps is a struct with fields XDir, YDir, XLim, YLim.
  xdir = axesCurrProps.XDir;
  ydir = axesCurrProps.YDir;
  if ~isfield(cache,'xlim'),
    xlim = axesCurrProps.XLim;
    ylim = axesCurrProps.YLim;
  else
    xlim = cache.xlim;
    ylim = cache.ylim;
  end
  axes_curr = struct('XLim',xlim, ...
                     'YLim',ylim, ...
                     'XDir',xdir', ...
                     'YDir',ydir, ...
                     'CameraViewAngleMode','auto') ;
end  % function
