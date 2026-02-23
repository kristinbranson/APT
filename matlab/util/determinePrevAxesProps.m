function result = determinePrevAxesProps(cache, axesCurrProps)
  % Returns a struct containing the desired display properties for the prev
  % axes, determined partly from cache and partly from the current axes
  % properties.  Does not mutate obj.
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
  result = struct('XLim',xlim, ...
                  'YLim',ylim, ...
                  'XDir',xdir', ...
                  'YDir',ydir, ...
                  'CameraViewAngleMode','auto') ;
end  % function
