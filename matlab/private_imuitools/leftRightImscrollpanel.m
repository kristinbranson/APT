function [hPanel,hImLeft,hImRight,hSpLeft,hSpRight] = ...
    leftRightImscrollpanel(parent,leftImage,rightImage)
%leftRightImscrollpanel Display two images side-by-side in scroll panels.
%   [hPanel,hImLeft,hImRight,hSpLeft,hSpRight] = ...
%      leftRightImscrollpanel(PARENT,leftImage,rightImage) displays 
%   leftImage and rightImage side-by-side each in its own scroll panel. 
%   PARENT is a handle to an object that can contain a uigridcontainer.
%
%   Arguments returned include:
%      hPanel   - Handle to panel containing two scroll panels
%      hImLeft  - Handle to left image object
%      hImRight - Handle to right image object
%      hSpLeft  - Handle to left scroll panel
%      hSpRight - Handle to right scroll panel
%
%   Example
%   -------
%       % Display two images side-by-side in scroll panels
%       left = imread('peppers.png');
%       right = edge(left(:,:,1),'canny');
%       hFig = figure('Toolbar','none',...
%                     'Menubar','none');
%       leftRightImscrollpanel(hFig,left,right);

%   Copyright 2005-2008 The MathWorks, Inc.
%   $Revision $  $Date: 2008/10/26 14:25:20 $

hFig = ancestor(parent,'figure');

% Call imageDisplayParseInputs twice
specificArgNames = {}; % No specific args needed
leftArgs = imageDisplayParseInputs(specificArgNames,leftImage);
rightArgs = imageDisplayParseInputs(specificArgNames,rightImage);


% Display left image
hAxLeft = axes('Parent',parent);
hImLeft = basicImageDisplay(hFig,hAxLeft,...
                            leftArgs.CData,leftArgs.CDataMapping,...
                            leftArgs.DisplayRange,leftArgs.Map,...
                            leftArgs.XData,leftArgs.YData);

% Display right image
hAxRight = axes('Parent',parent);
hImRight = basicImageDisplay(hFig,hAxRight,...
                            rightArgs.CData,rightArgs.CDataMapping,...
                            rightArgs.DisplayRange,rightArgs.Map,...
                            rightArgs.XData,rightArgs.YData);

% Create a scroll panel for left image
hSpLeft = imscrollpanel(parent,hImLeft);

% Create scroll panel for right image
hSpRight = imscrollpanel(parent,hImRight);

hPanel = uigridcontainer('v0',...
                         'parent',parent,...
                         'GridSize',[1 2],...
                         'Margin',1);
                     
hFig = iptancestor(hSpLeft,'Figure');
setChildColorToMatchParent([hSpLeft,hSpRight,hPanel],hFig);

% Reparent hSpLeft and hSpRight
set(hSpLeft,'parent',hPanel)
set(hSpRight,'parent',hPanel)

% Tag components for testing
set(hSpLeft,'Tag','LeftScrollPanel')
set(hSpRight, 'Tag','RightScrollPanel')  
