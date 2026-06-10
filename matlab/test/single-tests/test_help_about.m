function test_help_about()
% Test that Help > About works when no project is loaded.

[labeler, controller] = StartAPT('isInDebugMode', true) ;  %#ok<ASGLU>
cleanupObj = onCleanup(@()(delete(controller))) ;
drawnow('nocallbacks') ;

% Actuate Help > About
controller.menu_help_about_actuated_([], []) ;
drawnow('nocallbacks') ;
pos = controller.aboutFigure_.Position ;  %#ok<NASGU>  % round-trip to the host forces a sync
drawnow('nocallbacks') ;

% Find the About figure
aboutFigs = findall(groot, 'Type', 'figure', 'Name', 'About APT') ;
assert(~isempty(aboutFigs), ...
       'About dialog was not created.') ;
close(aboutFigs) ;

fprintf('test_help_about passed.\n') ;

end  % function
