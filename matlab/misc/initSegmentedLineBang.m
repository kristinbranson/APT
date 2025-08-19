function initSegmentedLineBang(hLine,xlim,onProps)
  % xlim: [2] line x-limits
  % yloc: [1] line y-loc
  % onProps: line P-V structure for on appearance
  
  assert(numel(xlim)==2);
  % assert(isscalar(yloc));
  assert(isstruct(onProps));

  x = [ xlim(1)-0.5:xlim(2)-0.5 ; ...
        xlim(1)+0.5:xlim(2)+0.5 ];
  x(end+1,:) = nan;
  y = nan(size(x));
  set(hLine,'XData',x(:),'YData',y(:));
  set(hLine,onProps);
  
  %obj.xLims = xlim;
  %obj.yLoc = yloc;     
end
