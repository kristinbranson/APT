function waitForFigureToSync(f)
% Block until the figure handle f's visual presentation is up-to-date with
% what has been specified.  Generally only needed for uifigure(), since
% they are more async by design.  Without this, can have a situation where
% user has done something that spawns a satellite figure, and the main
% figure cursor has gone from spinner to pointer *before* the satellite
% window actually appears.  So for some 100s of ms user is wondering what
% the heck happened, and where is the darn satellite window.  This fixes
% that.
%
% Implementation: first do the classic drawnow + Position round-trip dance
% to flush the host pipeline.  Then -- because uifigure layout is resolved
% asynchronously on the JS side and the figure's own Position can be
% reported correct while a child's pixel position is still the default --
% poll until the first child's rendered position has stabilized (two
% consecutive samples match and the position is no longer the
% [1 1 100 100] default), or until a 1-second safety timeout elapses.  For
% classic figures the position is correct on the first sample and the
% loop exits after a single pause.

drawnow('nocallbacks') ;
pos = f.Position ;  %#ok<NASGU>  % round-trip to the host forces a sync
drawnow('nocallbacks') ;

if isempty(f.Children)
  return
end
maxWait = 1 ;  % seconds
defaultPosition = [1 1 100 100] ;
deadline = tic() ;
previousPosition = [NaN NaN NaN NaN] ;
while toc(deadline) < maxWait
  childPosition = getpixelposition(f.Children(1)) ;
  isResolved = ~isequal(childPosition, defaultPosition) && ...
               childPosition(3) > 0 && childPosition(4) > 0 ;
  if isResolved && isequal(childPosition, previousPosition)
    break
  end
  previousPosition = childPosition ;
  pause(0.02) ;
  drawnow('nocallbacks') ;
end

end
