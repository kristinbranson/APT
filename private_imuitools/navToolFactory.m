function tools = navToolFactory(toolbar)
%navToolFactory Add navigational toolbar buttons to toolbar.
%   TOOLS = navToolFactory(TOOLBAR) returns TOOLS, a structure containing
%   handles to navigational tools. Tools for zoom in, zoom out, and pan are
%   added the TOOLBAR.
%
%   Note: navToolFactory does not set up callbacks for the tools.
%
%   Example
%   -------
%
%       hFig = figure('Toolbar','none',...
%                     'Menubar','none');
%       hIm = imshow('tissue.png'); 
%       hSP = imscrollpanel(hFig,hIm);
% 
%       toolbar = uitoolbar(hFig);
%       tools = navToolFactory(toolbar)
%
%   See also UITOGGLETOOL, UITOOLBAR.

%   Copyright 2005-2011 The MathWorks, Inc.  
%   $Revision: 1.1.6.3 $  $Date: 2011/08/09 17:55:32 $

[iconRoot, iconRootMATLAB] = ipticondir;

% Common properties
s.toolConstructor            = @uitoggletool;
s.properties.Parent          = toolbar;

% zoom in
s.iconConstructor            = @makeToolbarIconFromGIF;
s.iconRoot                   = iconRootMATLAB;    
s.icon                       = 'view_zoom_in.gif';
s.properties.TooltipString   = getString(message('images:imtoolUIString:zoomInTooltipString'));
s.properties.Tag             = 'zoom in toolbar button';
tools.zoomInTool = makeToolbarItem(s);

% zoom out
s.iconConstructor            = @makeToolbarIconFromGIF;
s.iconRoot                   = iconRootMATLAB;    
s.icon                       = 'view_zoom_out.gif';
s.properties.TooltipString   = getString(message('images:imtoolUIString:zoomOutTooltipString'));
s.properties.Tag             = 'zoom out toolbar button';
tools.zoomOutTool = makeToolbarItem(s);

% pan
s.iconConstructor            = @makeToolbarIconFromPNG;
s.iconRoot                   = iconRoot;    
s.icon                       = 'tool_hand.png';
s.properties.TooltipString   = getString(message('images:imtoolUIString:panTooltipString'));
s.properties.Tag             = 'pan toolbar button';
tools.panTool = makeToolbarItem(s);

