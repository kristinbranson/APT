function item = makeToolbarItem(s)
%makeToolbarItem Creates a toolbar item.
%   ITEM = makeToolbarItem(S) makes a toolbar item according the
%   specification in the structure S.
%
%   S must contain the following fields:
%  
%   S.toolConstructor - a function handle to call to create the tool
%   S.iconConstructor - a function hangle to call to create the icon
%   S.iconRoot        - path to icon file
%   S.icon            - name of icon file
%   S.properties      - a structure containing the parent toolbar, plus optional
%                       properties that will be passed directly to the toolConstructor
%
%   Example
%   -------
%       % Create a help toolbar button
%       hFig = figure('Toolbar','none');
%       toolbar =  uitoolbar(hFig);
%       [iconRoot, iconRootMATLAB] = ipticondir;
%
%       s = [];
%       s.toolConstructor            = @uipushtool;
%       s.iconConstructor            = @makeToolbarIconFromGIF;
%       s.iconRoot                   = iconRootMATLAB;    
%       s.icon                       = 'helpicon.gif';
%       s.properties.Parent          = toolbar;
%       s.properties.ClickedCallback = @(varargin) disp('help');
%       s.properties.ToolTip         = 'Help';
%  
%       item = makeToolbarItem(s);

%   See also UIPUSHTOOL, UITOGGLETOOL.

%   Copyright 2005-2006 The MathWorks, Inc.  
%   $Revision $  $Date $


  props = s.properties;
  props.CData = s.iconConstructor(fullfile(s.iconRoot,s.icon));

  item = s.toolConstructor(props);