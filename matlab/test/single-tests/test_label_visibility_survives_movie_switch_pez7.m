function test_label_visibility_survives_movie_switch_pez7()
% Test that label visibility is preserved when switching labeling modes.
%
% Regression test: switching from sequential to template and back caused
% labels to become invisible because doShowLabels state was lost when the
% LabelCoreModel was destroyed and recreated.

% Start APT with an SA project without trx (b/c only one animal)
projectFile = '/groups/branson/bransonlab/apt/unittest/pez7_al_updated_20241015.lbl' ;
[labeler, controller] = StartAPT('projfile', projectFile, ...
                                 'isInDebugMode', true, ...
                                 'isInYodaMode', true) ;
cleanupObj = onCleanup(@()(delete(controller))) ;
drawnow() ;

ax = controller.axes_curr ;

% Should start in template mode with labels visible
assert(labeler.labelMode == LabelMode.TEMPLATE, ...
       'Expected TEMPLATE mode at start, got %s', ...
       char(labeler.labelMode)) ;

% Record the initial label point positions
[xDataBefore, yDataBefore] = getLabelPointXYData(ax) ;
assert(~isempty(xDataBefore), 'No label-point graphics found at start.') ;

% Switch to sequential mode via menu
controller.controlActuated('menu_label_sequential_mode') ;
drawnow() ;

assert(labeler.labelMode == LabelMode.SEQUENTIAL, ...
       'Expected SEQUENTIAL mode after switch, got %s', ...
       char(labeler.labelMode)) ;

% Switch back to template mode via menu
controller.controlActuated('menu_label_template_mode') ;
drawnow() ;

assert(labeler.labelMode == LabelMode.TEMPLATE, ...
       'Expected TEMPLATE mode after switching back, got %s', ...
       char(labeler.labelMode)) ;

% Verify label points have the same positions as before the mode switches
[xDataAfter, yDataAfter] = getLabelPointXYData(ax) ;
assert(isequal(xDataBefore, xDataAfter), ...
       'Label point XData changed after mode switch round-trip.') ;
assert(isequal(yDataBefore, yDataAfter), ...
       'Label point YData changed after mode switch round-trip.') ;

% Switch to movie 9 (which has no labels) and verify template points are visible
labeler.movieSet(9) ;
drawnow() ;

[xDataMov9, yDataMov9] = getLabelPointXYData(ax) ;
assert(~isempty(xDataMov9), ...
       'No label-point graphics found after switching to unlabeled movie 9.') ;
assert(~all(isnan(xDataMov9)), ...
       'Template points have non-finite XData on unlabeled movie 9.') ;
assert(~all(isnan(yDataMov9)), ...
       'Template points have non-finite YData on unlabeled movie 9.') ;

fprintf('%s passed.\n', mfilename()) ;

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
