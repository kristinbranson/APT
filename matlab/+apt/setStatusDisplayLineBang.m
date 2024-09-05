function setStatusDisplayLineBang(hfig, str, isallgood)  
  % Set either or both of the status message line and the color of the status
  % message.  Any of the two (non-hfig) args can be empty, in which case that
  % aspect is not changed.  hfig's guidata must have a text_clusterstatus field
  % containing the handle of an 'text' appropriate graphics object.
  
  handles = guidata(hfig);
  text_h = handles.text_clusterstatus ;
  if ~exist('str', 'var') ,
    str = [] ;
  end
  if ~exist('isallgood', 'var') ,
    isallgood = [] ;
  end
  if isempty(str) ,
    % do nothing
  else
    set(text_h, 'String', str) ;
  end
  if isempty(isallgood) ,
    % do nothing    
  else
    color = fif(isallgood, 'g', 'r') ;
    set(text_h, 'ForegroundColor',color) ;
  end
  drawnow('limitrate', 'nocallbacks') ;
end  % function    
