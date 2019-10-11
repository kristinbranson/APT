function printImageToFigure(hParent)
%PRINTIMAGETOFIGURE Print axes contents to a figure.
%   PRINTIMAGETOFIGURE(HPARENT) prints the contents of an axes parented by
%   HPARENT to a new figure.  The new figure window is centered on the
%   screen.  All objects in the axes are copied to the new figure but all
%   interactive behaviors are removed from the copied objects.
%
%   HPARENT must either be a figure or an instance of IMSCROLLPANEL.
%
%   Copyright 2004-2008 The MathWorks, Inc.
%   $Revision: 1.1.8.6 $  $Date: 2010/09/13 16:14:23 $

if strcmpi(get(hParent,'Type'),'uipanel')
    % parent is imscrollpanel
    hScrollpanel = hParent;
    old_fig = ancestor(hScrollpanel,'figure');
    old_axes = findobj(old_fig, 'type', 'axes');

    scrollpanelAPI = iptgetapi(hScrollpanel);
    mag  = scrollpanelAPI.getMagnification();
    vis_rect = scrollpanelAPI.getVisibleImageRect();

    fig_width  = mag * vis_rect(3);
    fig_height = mag * vis_rect(4);
    xlim = vis_rect(1) + [0 vis_rect(3)];
    ylim = vis_rect(2) + [0 vis_rect(4)];

else
    % parent is a figure
    old_fig = hParent;
    old_axes = findobj(old_fig, 'type', 'axes');
    old_fig_pos = get(hParent,'Position');

    fig_width = old_fig_pos(3);
    fig_height = old_fig_pos(4);
    xlim = get(old_axes,'Xlim');
    ylim = get(old_axes,'Ylim');

end

% compute properties of new figure
fp = figparams;
fig_left   = round((fp.ScreenWidth - fig_width) / 2);
fig_bottom = round((fp.ScreenHeight - fig_height) / 2);
fig_position = [fig_left fig_bottom fig_width fig_height];
old_cmap = get(old_fig,'Colormap');

% create new figure
h_figure = figure('Visible', 'off', ...
    'Units', 'pixels', ...
    'Position', fig_position, ...
    'Colormap', old_cmap,...
    'PaperPositionMode', 'auto', ...
    'InvertHardcopy', 'off',...
    'Tag','printImageToFigure');

% copy all objects contained in the axes and reset some properties
h_axes = copyobj(old_axes, h_figure);
set(h_axes, 'Units', 'normalized', ...
    'Position', [0 0 1 1], ...
    'XLim', xlim,...
    'YLim', ylim);

% remove interactive behaviors from copied objects
all_objects = findall(h_axes);

% The value expected for the UIContextMenu property of an object is a
% handle in HG2 vs. a double handle in HG1.
if feature('HGUsingMATLABClasses')
    set(all_objects,'UIContextMenu',handle.empty)
else
    set(all_objects,'UIContextMenu','');
end

set(all_objects,'ButtonDownFcn','');
set(all_objects,'DeleteFcn','');
set(all_objects,'UserData','');
set(all_objects,'HitTest','off');

% remove all appdata
removeAppData(findall(h_figure));

% display the new figure;
set(h_figure, 'Visible', 'on');


%--------------------------------
function removeAppData(obj_array)

for i = 1:numel(obj_array)
    obj = obj_array(i);
    
    cur_data = getappdata(obj);
    if ~isempty(cur_data)
        data_names = fields(cur_data);
        cellfun(@(field_name) rmappdata(obj,field_name),data_names);
    end
end
