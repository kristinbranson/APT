function cmenu = createROIContextMenu(h_fig,getPosition,setColor)
%createROIContextMenu creates a context menu for setting color and copying position.
%    CMENU = createROIContextMenu(CMENU_OBJ,getPosition,setColor) creates a
%    context menu which is parented to H_FIG. getPosition and setColor are
%    function handles provided by the client. The output argument CMENU is
%    an HG handle to a uicontextmenu.

%   Copyright 2007-2008 The MathWorks, Inc.    
    
cmenu = uicontextmenu('Parent',h_fig);

% Create context menu item for copying position to clipboard
uimenu(cmenu, ...
	'Label', getString(message('images:roiContextMenuUIString:copyPositionContextMenuLabel')), ...
	'Tag', 'copy position cmenu item',...
	'Callback', @copyPosition )

% Create context menu item for setting color
set_color_menu = uimenu(cmenu, ...
	'Label', getString(message('images:roiContextMenuUIString:setColorContextMenuLabel')), ...
	'Tag','set color cmenu item');

color_choices = iptui.getColorChoices();
color_submenus = zeros(1,numel(color_choices));
for k = 1:numel(color_choices)
	color_submenus(k) = uimenu(set_color_menu,...
		'Label', color_choices(k).Label, ...
		'Tag', color_choices(k).Tag,...
		'Callback', @(varargin) setColor(color_choices(k).Color));
end

% Check first color choice in context menu
set(color_submenus(1),'Checked','on');
% Initialize ROI to first color choice in context menu
setColor(color_choices(1).Color);

    %------------------------------
	function copyPosition(varargin)
		clipboard('copy',getPosition())
	end %copyPosition
    
end %createROIContextMenu
