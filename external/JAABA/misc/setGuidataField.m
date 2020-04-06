function setGuidataField(fig,fieldName,x)
% Sets a single field from fig's guidata(), assumed to be a scalar struct,
% to the value x.

g=guidata(fig);
g.(fieldName)=x;
guidata(fig,g);

end
