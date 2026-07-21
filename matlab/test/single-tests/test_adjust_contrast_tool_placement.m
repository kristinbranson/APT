function test_adjust_contrast_tool_placement()
% Test that the Adjust Contrast tool opens at its aligned position.
%
% imcontrast_kb() aligns the tool window to the target figure with
% iptwindowalign() (left edges aligned, tool top at the target figure
% bottom, clamped to stay onscreen).  On >=R2025a, figure windows are
% placed by the OS window manager when first shown: a position set
% while the figure is still invisible is not honored, so the tool
% materialized wherever the window manager chose -- typically centered
% over, and stacked behind, the main APT window.  The net effect was
% that View > Adjust Brightness/Contrast... appeared to do nothing.
%
% The desired behavior is that after the menu actuation the tool
% window already sits at its aligned position.  We check this by
% re-running the alignment and asserting that the tool does not move.
%
% Note: this test is only meaningful in a MATLAB session with a real
% display.  Without one, window placement is trivially honored and the
% test passes even in the presence of the bug.

% Start APT with an SA project (single view, so no view-picker dialog)
linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/pez7_al_updated_20241015.lbl' ;
[projectFile, replace_path] = localize_test_project_path(linux_project_file_path) ;
[labeler, controller] = StartAPT('projfile', projectFile, ...
                                 'replace_path', replace_path, ...
                                 'isInDebugMode', true, ...
                                 'isInYodaMode', true) ;  %#ok<ASGLU>
cleanupObj = onCleanup(@()(delete(controller))) ;
drawnow() ;

% Open the Adjust Contrast tool via the menu
controller.controlActuated('menu_view_adjustbrightness') ;
drawnow() ;
pause(2) ;  % give the window manager time to place the window

toolFigure = findall(groot(), 'Type', 'figure', 'Tag', 'imcontrast') ;
assert(isscalar(toolFigure), 'Adjust Contrast tool figure not found') ;
cleanupToolObj = onCleanup(@()(delete(toolFigure))) ;

mainFigure = ancestor(controller.axes_curr, 'figure') ;
positionBefore = get(toolFigure, 'OuterPosition') ;

% Re-run the alignment that imcontrast_kb() is supposed to have
% applied already.  If the tool is where it should be, this is a no-op.
iptwindowalign(mainFigure, 'left', toolFigure, 'left') ;
iptwindowalign(mainFigure, 'bottom', toolFigure, 'top') ;
drawnow() ;
pause(1) ;
positionAfter = get(toolFigure, 'OuterPosition') ;

offset = max(abs(positionAfter - positionBefore)) ;
assert(offset < 50, ...
       'Adjust Contrast tool was not at its aligned position: re-aligning moved it by %d pixels', ...
       round(offset)) ;
end  % function
