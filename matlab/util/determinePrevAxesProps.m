function result = determinePrevAxesProps(cache, currAxesProps)
  % Returns a struct containing the desired display properties for the prev
  % axes, determined partly from cache and partly from the current axes
  % properties.  Does not mutate obj.
  % cache is a PrevAxesTargetCache.
  % axesCurrProps is a struct with fields XDir, YDir, XLim, YLim.
  xdir = currAxesProps.XDir;
  ydir = currAxesProps.YDir;
  if isempty(cache.xlim),
    xlim = currAxesProps.XLim;
    ylim = currAxesProps.YLim;
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
