function setStringAndFitWidthBang(hTxt, str)
% Set the String of a text uicontrol and adjust its width to fit.
hTxt.String = str ;
drawnow() ;
extent = hTxt.Extent ;
hTxt.Position(3) = extent(3) + 2 ;  % the + 2 is a fudge factor
end  % function
