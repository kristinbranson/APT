function test_label_visibility_survives_mode_switch()
% Test that label visibility is preserved when switching labeling modes.
%
% Regression test: switching from sequential to template and back caused
% labels to become invisible because doShowLabels state was lost when the
% LabelCoreModel was destroyed and recreated.

% Start APT with an SA project that has trx
projectFile = '/groups/branson/bransonlab/apt/unittest/with-trx-project.lbl' ;
[labeler, controller] = StartAPT('projfile', projectFile, ...
                                 'isInDebugMode', true, ...
                                 'isInYodaMode', true) ;
cleanupObj = onCleanup(@()(delete(controller))) ;
drawnow() ;

ax = controller.axes_curr ;

% Should start in sequential mode with labels visible
assert(labeler.labelMode == LabelMode.SEQUENTIAL, ...
       'Expected SEQUENTIAL mode at start, got %s', ...
       char(labeler.labelMode)) ;

% Record the initial label point positions
[xDataBefore, yDataBefore] = getLabelPointXYData(ax) ;
assert(~isempty(xDataBefore), 'No label-point graphics found at start.') ;

% Switch to template mode via menu
controller.controlActuated('menu_setup_template_mode') ;
drawnow() ;

assert(labeler.labelMode == LabelMode.TEMPLATE, ...
       'Expected TEMPLATE mode after switch, got %s', ...
       char(labeler.labelMode)) ;

% Switch back to sequential mode via menu
controller.controlActuated('menu_setup_sequential_mode') ;
drawnow() ;

assert(labeler.labelMode == LabelMode.SEQUENTIAL, ...
       'Expected SEQUENTIAL mode after switching back, got %s', ...
       char(labeler.labelMode)) ;

% Verify label points have the same positions as before the mode switches
[xDataAfter, yDataAfter] = getLabelPointXYData(ax) ;
assert(isequal(xDataBefore, xDataAfter), ...
       'Label point XData changed after mode switch round-trip.') ;
assert(isequal(yDataBefore, yDataAfter), ...
       'Label point YData changed after mode switch round-trip.') ;

fprintf('test_label_visibility_survives_mode_switch passed.\n') ;

end  % function


function [xData, yData] = getLabelPointXYData(ax)
% Get sorted XData and YData arrays from label-point line objects on the axes.
children = get(ax, 'Children') ;
tags = get(children, 'Tag') ;
tfIsLabelPt = startsWith(tags, 'LabelCore_Pts_') ;
hLabelPts = children(tfIsLabelPt) ;
% Keep only line objects (not text objects, which share the same tag prefix)
tfIsLine = arrayfun(@(h) isa(h, 'matlab.graphics.chart.primitive.Line'), hLabelPts) ;
hLines = hLabelPts(tfIsLine) ;
% Sort by tag so order is deterministic
lineTags = get(hLines, 'Tag') ;
[~, sortOrder] = sort(lineTags) ;
hLines = hLines(sortOrder) ;
xData = arrayfun(@(h) h.XData, hLines) ;
yData = arrayfun(@(h) h.YData, hLines) ;
end  % function
