function y=getGuidataField(fig,fieldName)
% gets a single field from fig's guidata(), assumed to be a scalar struct

g=guidata(fig);
y=g.(fieldName);

end
