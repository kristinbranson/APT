function widgetAPI = createClimWindowOnAxes(hAx,clim,maxCounts)
%createClimWindowOnAxes Create draggable window in imcontrast tool
%   widgetAPI = createClimWindowOnAxes(hAx,clim,maxCounts) creates a draggable
%   window on the axes specified by the handle HAX.
%
%   This is used by IMCONTRAST.

%   Copyright 2005-2008 The MathWorks, Inc.
%   $Revision: 1.1.6.5 $  $Date: 2010/06/07 16:32:29 $

hPatch = patch([clim(1) clim(1) clim(2) clim(2)], ...
    [0 maxCounts maxCounts 0], [1 0.8 0.8], ...
    'parent', hAx, ...
    'zData', [-2 -2 -2 -2], ...
    'tag', 'window patch');

% There is a drawing stacking bug (g298614) with the painters renderer.
% Workaround this by using uistack to ensure patch window is below stem
% lines.
uistack(hPatch,'bottom');

hMinLine = line('parent', hAx, ...
    'tag', 'min line', ...
    'xdata', [clim(1) clim(1)], ...
    'ydata', [0 maxCounts], ...
    'ZData', [-1 -1], ...
    'color', [1 0 0], ...
    'LineWidth', 1);

hMaxLine = line('parent', hAx, ...
    'tag', 'max line', ...
    'xdata', [clim(2) clim(2)], ...
    'ydata', [0 maxCounts], ...
    'ZData', [-1 -1], ...
    'color', [1 0 0], ...
    'LineWidth', 1);

[width, center] = computeWindow(clim);
hCenterLine = line('parent', hAx, ...
    'tag', 'center line', ...
    'xdata', [center center], ...
    'ydata', [0 maxCounts], ...
    'zdata', [-2 -2], ...
    'color', [1 0 0], ...
    'LineWidth', 1, ...
    'LineStyle', '--');

% Add handles to make moving the endpoints easier for very small windows.
[XShape, YShape] = getSidePatchShape;
XLim = get(hAx, 'XLim');
YLim = get(hAx, 'YLim');

hMinPatch = patch('parent', hAx, ...
    'XData', clim(1) - (XShape * double(XLim(2) - XLim(1))), ...
    'YData', YShape * YLim(2), ...
    'ZData', ones(size(XShape)), ...
    'FaceColor', [1 0 0], ...
    'EdgeColor', [1 0 0], ...
    'tag', 'min patch');

hMaxPatch = patch('parent', hAx, ...
    'XData', clim(2) + (XShape * double(XLim(2) - XLim(1))), ...
    'YData', YShape * YLim(2), ...
    'ZData', ones(size(XShape)), ...
    'FaceColor', [1 0 0], ...
    'EdgeColor', [1 0 0], ...
    'tag', 'max patch');

[XShape, YShape] = getTopPatchShape;
hCenterPatch = patch('parent', hAx, ...
    'XData', center + XShape .* double(XLim(2) - XLim(1)), ...
    'YData', YShape * (YLim(2) - YLim(1)), ...
    'ZData', ones(size(XShape)), ...
    'FaceColor', [1 0 0], ...
    'EdgeColor', [1 0 0], ...
    'tag', 'center patch');

createWidgetAPI;

    %=======================
    function createWidgetAPI

        widgetAPI.centerLine.handle = hCenterLine;
        widgetAPI.centerLine.get    = @() getXLocation(hCenterLine);
        widgetAPI.centerLine.set    = @setCenterLine;

        widgetAPI.centerPatch.handle = hCenterPatch;
        widgetAPI.centerPatch.get    = @() getXLocation(hCenterPatch);
        widgetAPI.centerPatch.set    = @setCenterPatch;

        widgetAPI.maxLine.handle = hMaxLine;
        widgetAPI.maxLine.get    = @() getXLocation(hMaxLine);
        widgetAPI.maxLine.set    = @(clim) setXLocation(hMaxLine,clim(2));

        widgetAPI.minLine.handle = hMinLine;
        widgetAPI.minLine.get    = @() getXLocation(hMinLine);
        widgetAPI.minLine.set    = @(clim) setXLocation(hMinLine,clim(1));

        widgetAPI.maxPatch.handle = hMaxPatch;
        widgetAPI.maxPatch.get    = @() getXLocation(hMaxPatch);
        widgetAPI.maxPatch.set    = @setMaxPatch;

        widgetAPI.minPatch.handle = hMinPatch;
        widgetAPI.minPatch.get    = @() getXLocation(hMinPatch);
        widgetAPI.minPatch.set    = @setMinPatch;

        widgetAPI.bigPatch.handle = hPatch;
        widgetAPI.bigPatch.get    = @() getXLocation(hPatch);
        widgetAPI.bigPatch.set    = @setPatch;

        %==========================
        function setCenterLine(clim)
            [width,center] = computeWindow(clim);
            setXLocation(hCenterLine,center);
        end

        %==========================
        function setCenterPatch(clim)
            [width,center] = computeWindow(clim);
            topPatchXData = getTopPatchShape * double(getPatchScale);
            set(hCenterPatch, 'XData', center + topPatchXData);
        end
        %==========================
        function setMaxPatch(clim)
            sidePatchXData = getSidePatchShape * double(getPatchScale);
            set(hMaxPatch, 'XData', clim(2) + sidePatchXData);
        end

        %==========================
        function setMinPatch(clim)
            sidePatchXData = getSidePatchShape * double(getPatchScale);
            set(hMinPatch, 'XData', clim(1) - sidePatchXData);
        end

        %==========================
        function setPatch(clim)
            set(hPatch, 'XData', [clim(1) clim(1) clim(2) clim(2)]);
        end

        %===========================
        function value = getXLocation(h)
            value = get(h,'xdata');
            value = value(1);
        end

        %========================
        function setXLocation(h,value)
        % these are the same because we are setting the location of a 
        % vertical line
            set(h,'XData',[value value]);
        end

        %==========================================
        function [xFactor, yFactor] = getPatchScale
            xFactor = XLim(2) - XLim(1);
            yFactor = YLim(2) - YLim(1);
        end
    end %createWidgetAPI
end %createClimWindowOnAxes

%==========================================================================
function [width, center] = computeWindow(CLim)
width = CLim(2) - CLim(1);
center = CLim(1) + width ./ 2;
end

%==========================================================================
function [XData, YData] = getSidePatchShape
XData = [0.00 -0.007 -0.007 0.00 0.01 0.02 0.02 0.01];
YData = [0.40  0.42   0.58  0.60 0.60 0.56 0.44 0.40];
end

%==========================================================================
function [XData, YData] = getTopPatchShape
XData = [-0.015 0.015 0];
YData = [1 1 0.95];
end



