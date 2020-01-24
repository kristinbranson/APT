function varargout = setChildColorToMatchParent(child,parent)
%setChildColorToMatchParent Sets color of child to match parent's color.
%   setChildColorToMatchParent(child,parent) matches the child's
%   BackgroundColor property to the parent's color. The parent may be a
%   figure, uipanel, or uicontainer. The child is a uipanel or uicontainer.
%
%   COLOR = setChildColorToMatchParent(...) returns the parent's background
%   color.
  
%   Copyright 1993-2004 The MathWorks, Inc.  
%   $Revision $  $Date: 2004/08/10 01:50:35 $

if strcmp(get(parent,'type'),'figure')
    background = get(parent,'Color');
else
    background = get(parent,'BackgroundColor');       
end

set(child,'BackgroundColor',background);       

if nargout > 0
    varargout{1} = background;
end
