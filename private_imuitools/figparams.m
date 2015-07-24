function out = figparams
%FIGPARAMS Calculate figure parameters.  
%   OUT = FIGPARAMS calculates absolute figure parameters that can be
%   used in laying out axes positions.
%
%   The parameters that determine these dimensions only need to be computed
%   once per MATLAB session because they are independent of the image being
%   displayed and depend on the machine running the display function.

%   Copyright 1993-2011 The MathWorks, Inc.
%   $Revision: 1.1.8.3 $  $Date: 2011/02/09 18:56:18 $


% lock file as this should only be calculated once per MATLAB session
mlock

persistent p
if ~isempty(p)
    out = p;
    return
end

hfig = figure('Visible', 'off', 'IntegerHandle', 'off', 'Units', 'pixels');
hax  = axes('Parent', hfig, 'Units', 'pixels', ...
            'XAxisLocation', 'bottom', 'YAxisLocation', 'left');

ax_pos  = get(hax, 'Position');
fig_pos = get(hfig, 'Position');
fig_outer_pos = get(hfig, 'OuterPosition');

% Decorations are the titlebar, toolbars, menu bars, and the edges of
% figure windows. Really, just the difference between the 'Position' and
% the 'OuterPosition.'
%
% Note: The calculation of the 'OuterPosition' is platform dependent and
% there have been bugs on certain platforms.
p.LeftDecoration   = abs(fig_pos(3) - fig_outer_pos(3))/2;
p.BottomDecoration = p.LeftDecoration;
p.RightDecoration  = p.LeftDecoration;
p.TopDecoration    = fig_outer_pos(4) - fig_pos(4) - p.BottomDecoration + ...
    60; % fudge factor to account for menus and toolbar which seem to not
        % be part of figure OuterPosition.

set(hax,'YLim',[1 1000]); % So tick labels will fit for images with 
                          % 1000s of pixels in either dimension.

htitle  = get(hax, 'title');
hxlabel = get(hax, 'xlabel');
hylabel = get(hax, 'ylabel');

set([htitle hxlabel hylabel], 'Units', 'pixels', 'String', 'X');

% premultipliers are tuned to "look good" for most image sizes
title_extent = get(htitle, 'Extent');
p.AxesTitleHeight = 2 * (title_extent(2) + title_extent(4)/2 - ax_pos(4));

xlabel_extent = get(hxlabel, 'Extent');
p.XLabelHeight = -2 * (xlabel_extent(2) + xlabel_extent(4)/2);

ylabel_extent = get(hylabel, 'Extent');
p.YLabelWidth = -1.8 * (ylabel_extent(1) + ylabel_extent(3)/2);

close(hfig);

screen_units = get(0, 'Units');
set(0, 'Units', 'pixels');
screen_size = get(0, 'ScreenSize');
set(0, 'Units', screen_units);

if any(screen_size(3:4) <= 1)
    p.ScreenWidth  = 10000;
    p.ScreenHeight = 10000;
else
    p.ScreenWidth = screen_size(3);
    p.ScreenHeight = screen_size(4);
end

% pre-compute total decorations
p.horizontalDecorations = p.RightDecoration  + p.LeftDecoration;
p.verticalDecorations   = p.BottomDecoration + p.TopDecoration;

% pre-compute loose border sizes (gutterWidth and gutterHeight)
p.looseBorderWidth = 2 * ceil(p.YLabelWidth);
p.looseBorderHeight = ceil(p.XLabelHeight + p.AxesTitleHeight);

out = p;
