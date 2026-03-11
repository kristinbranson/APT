function test_help_about()
% Test that Help > About works when no project is loaded.

[labeler, controller] = StartAPT('isInDebugMode', true) ;  %#ok<ASGLU>
cleanupObj = onCleanup(@()(delete(controller))) ;
drawnow() ;

% Actuate Help > About
controller.menu_help_about_actuated_([], []) ;
drawnow() ;

% Find the About figure
aboutFigs = findall(groot, 'Type', 'figure', 'Name', 'About') ;
assert(~isempty(aboutFigs), ...
       'About dialog was not created.') ;
close(aboutFigs) ;

fprintf('test_help_about passed.\n') ;

end  % function
