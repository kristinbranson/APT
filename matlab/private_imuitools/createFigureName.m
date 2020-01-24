function figName = createFigureName(toolName,targetFigureHandle)
% CREATEFIGURENAME(TOOLNAME, TARGETFIGUREHANDLE) creates a name for the figure
% created by the tool, TOOLNAME.  The figure name, FIGNAME, will include
% TOOLNAME and the name of the figure on which the tool depends. TOOLNAME must
% be a string, and TARGETFIGUREHANDLE must be a valid handle to the figure on
% which TOOLNAME depends.
%
%   Example
%   -------
%       h = imshow('bag.png');
%       hFig = figure;
%       imhist(imread('bag.png'));
%       toolName = 'Histogram';
%       targetFigureHandle = ancestor(h,'Figure');
%       name = createFigureName(toolName,targetFigureHandle);
%       set(hFig,'Name',name);
%
%   See also IMAGEINFO, BASICIMAGEINFO, IMPIXELREGION.

%   Copyright 1993-2010 The MathWorks, Inc.
%   $Revision: 1.1.8.8 $  $Date: 2011/08/09 17:55:22 $
  
  
if ~ischar(toolName)
  error(message('images:createFigureName:invalidInput'))
end

if ishghandle(targetFigureHandle,'figure')

  targetFigureName = get(targetFigureHandle,'Name');
  
  if isempty(targetFigureName) && isequal(get(targetFigureHandle, ...
                                              'IntegerHandle'), 'on')
    targetFigureName = getString(message('images:commonUIString:createFigureNameEmptyName',...
                                         double(targetFigureHandle)));
  end

  if ~isempty(targetFigureName)
    figName = sprintf('%s (%s)', toolName, targetFigureName);
  else
    figName = toolName;
  end
  
else
  error(message('images:createFigureName:invalidFigureHandle'))
end

  
