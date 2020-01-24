function [hPanel,hImOvLeft,hImOvRight] = leftRightImoverviewpanel(parent,hImLeft,hImRight)
%leftRightImoverviewpanel Display two images side-by-side in overview panels.
%   [hPanel,hOvLeft,hOvRight] = ...
%      leftRightImoverviewpanel(PARENT,hImLeft,hImRight) displays side-by-side
%   overview panels associated with hImLeft and hImRight. The handles hImLeft
%   and hImRight must refer to image objects that are each in and image scroll
%   panel. PARENT is a handle to an object that can contain a uigridcontainer.
%
%   Arguments returned include:
%      hPanel     - Handle to panel containing two overview panels
%      hImOvLeft  - Handle to image in left overview panel
%      hImOvRight - Handle to image in right overview panel
%
%   Example
%   -------
%       % Display two images side-by-side in scroll panels,
%       % and create overview panels beneath the scroll panels.
%       left = imread('peppers.png');
%       right = edge(left(:,:,1),'canny');
%       hFig = figure('Toolbar','none',...
%                     'Menubar','none');
%       [hSP,hImL,hImR] = leftRightImscrollpanel(hFig,left,right);
%       hOP = leftRightImoverviewpanel(hFig,hImL,hImR);
%       set(hSP,'Position',[0 .5 1 .5])
%       set(hOP,'Position',[0 0 1 .5])

%   Copyright 2005-2008 The MathWorks, Inc.
%   $Revision $  $Date: 2008/04/03 03:12:55 $
  
% Create an overview panel for left image
hOvLeft = imoverviewpanel(parent,hImLeft);

% Create an overview panel for right image
hOvRight = imoverviewpanel(parent,hImRight);

set([hOvLeft hOvRight],'BorderType','etchedin')

hPanel = uigridcontainer('v0',...
                         'parent',parent,...
                         'GridSize',[1 2],...
                         'Margin',1);
                     
hFig = iptancestor(hOvLeft,'Figure');
setChildColorToMatchParent([hOvLeft,hOvRight,hPanel],hFig);
                     
% Reparent hOvLeft and hOvRight
set(hOvLeft,'parent',hPanel)
set(hOvRight,'parent',hPanel)

% Tag components for testing
set(hOvLeft,'Tag','LeftOverviewPanel')
set(hOvRight, 'Tag','RightOverviewPanel')  

hImOvLeft  = findobj(hOvLeft,'Type','image');
hImOvRight = findobj(hOvRight,'Type','image');

