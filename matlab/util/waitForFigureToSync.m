function waitForFigureToSync(f)
% Block until the figure handle f's visual presentation is up-to-date with
% what has been specified.  Generally only needed for uifigure(), since
% they are more async by design.  Without this, can have a situation where
% user has done something that spawns a satellite figure, and the main
% figure cursor has gone from spinner to pointer *before* the satellite
% window actually appears.  So for some 100s of ms user is wondering what
% the heck happened, and where is the darn satellite window.  This fixes
% that.

drawnow('nocallbacks') ;
pos = f.Position ;  %#ok<NASGU>  % round-trip to the host forces a sync
drawnow('nocallbacks') ;

end
