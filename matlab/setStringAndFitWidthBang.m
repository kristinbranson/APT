function setStringAndFitWidthBang(hText, str)
% Set the String of a text uicontrol and adjust its width to fit.
hText.String = str ;
shrinkWrapTextControlBang(hText) ;
end  % function
