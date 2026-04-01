function shrinkWrapTextControlBang(hText)
% Set the width of a text uicontrol to just fit its text.
% The side that stays fixed depends on HorizontalAlignment:
%   'left'   => left side fixed
%   'right'  => right side fixed
%   'center' => center fixed

padding = 4 ;
oldPosition = hText.Position ;
drawnow('nocallbacks') ;  % Have to do this to make sure Extent is up-to-date
extent = hText.Extent ;
newWidth = extent(3) + padding ;
alignment = get(hText, 'HorizontalAlignment') ;
oldX = oldPosition(1) ;
oldWidth = oldPosition(3) ;
switch alignment
  case 'left'
    newX = oldX ;
  case 'right'
    newX = oldX + oldWidth - newWidth ;
  case 'center'
    newX = oldX + oldWidth / 2 - newWidth / 2 ;
  otherwise
    error('Unexpected HorizontalAlignment: %s', alignment) ;
end  % switch
hText.Position = replaceAt(oldPosition, [1 3], [newX newWidth]) ;

end  % function
