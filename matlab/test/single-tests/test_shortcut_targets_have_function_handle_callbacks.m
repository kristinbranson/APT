function test_shortcut_targets_have_function_handle_callbacks()
  % Every keyboard-shortcut target in the main window carries a function
  % handle as its Callback (or MenuSelectedFcn).  The two key handlers,
  % LabelerController.cbkKPF() and the multiview core's kpf(), dispatch a
  % shortcut by calling that handle.  They used to carry an eval()/evalin()
  % branch for a char Callback, which nothing in APT sets:
  % set_standard_callback_if_none_bang() installs a handle on every control.
  % This pins that premise, so the char branch can stay gone.

  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;

  [labeler, controller] = ...
    StartAPT('projfile', project_file_path, ...
             'replace_path', replace_path) ;
  cleaner = onCleanup(@()(delete(controller))) ;
  drawnow() ;

  shortcuts = labeler.getShortcuts() ;
  tags = fieldnames(shortcuts) ;
  assert(~isempty(tags), 'The project has no keyboard shortcuts, so the test would be vacuous') ;

  % A shortcut may name a control that no longer exists; cbkKPF() tolerates
  % that, so skip those, but every target that does exist must hold a handle.
  mainFigure = controller.mainFigure_ ;
  foundCount = 0 ;
  for i = 1 : numel(tags)
    tag = tags{i} ;
    control = findobj(mainFigure, 'Tag', tag) ;
    if isempty(control)
      continue
    end
    assert(isscalar(control), 'Shortcut target %s names %d objects in the main window', tag, numel(control)) ;
    foundCount = foundCount + 1 ;
    if isprop(control, 'Callback')
      callback = control.Callback ;
    elseif isprop(control, 'MenuSelectedFcn')
      callback = control.MenuSelectedFcn ;
    else
      error('Shortcut target %s has neither a Callback nor a MenuSelectedFcn', tag) ;
    end
    assert(isa(callback, 'function_handle'), ...
           'Shortcut target %s has a %s callback, not a function handle', tag, class(callback)) ;
  end
  assert(foundCount > 0, 'None of the %d shortcut targets exists in the main window', numel(tags)) ;

  % A Ctrl+key event for a shortcut resolves to its tag, which is the path
  % the multiview core's key handler takes before calling the handle.
  firstTag = tags{1} ;
  keyEvent = struct('Key', shortcuts.(firstTag), 'Character', '') ;
  keyEvent.Modifier = {'control'} ;
  match = controller.matchShortcut(keyEvent) ;
  assert(~isempty(match) && any(strcmp(match(:, 1), firstTag)), ...
         'Ctrl+%s did not match shortcut %s', shortcuts.(firstTag), firstTag) ;
end  % function
