function setSegmentedLineOnOffAtBang(hLine,x,tf)
  % Set segmented line on/off at location(s) x
  %
  % x: positive integer vector of coords/locs
  yLoc = 1 ;
  
  if isempty(x)
    return;
  end
  x = x(:);
    
  % assert(all(obj.xLims(1)<=x & x<=obj.xLims(2)));
  
  xx = (x-1)*3;
  yDataIdx = [xx+1;xx+2];
  if tf
    hLine.YData(yDataIdx) = yLoc;
  else
    hLine.YData(yDataIdx) = nan;
  end
end
