function test_shortcuts_dialog()
  % Test that the Edit Shortcuts dialog opens for a project with a label core.
  %
  % Regression test: ShortcutsDialog asked the label core *model* for
  % LabelShortcuts(), which has lived on LabelCoreController since LabelCore
  % was split into a model and a controller, so File > Edit Shortcuts...
  % errored for any project that has a label core.

  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;

  [labeler, controller] = ...
    StartAPT('projfile', project_file_path, ...
             'replace_path', replace_path) ;
  cleaner = onCleanup(@()(delete(controller))) ;
  drawnow() ;

  % Both of these would make the test vacuous, so check them up front.
  assert(~isempty(labeler.lblCore), ...
         'Test project did not create a label core.') ;
  fixedShortcuts = controller.lblCoreController_.LabelShortcuts() ;
  assert(~isempty(fixedShortcuts), ...
         'The label core reports no fixed shortcuts.') ;

  % File > Edit Shortcuts... blocks in uiwait(), so build the dialog directly.
  shortcutsFigure = ShortcutsDialog(controller) ;
  dialogCleaner = onCleanup(@()(delete(shortcutsFigure))) ;
  drawnow() ;

  assert(isgraphics(shortcutsFigure, 'figure'), ...
         'The Shortcuts dialog did not create a valid figure.') ;

  % The dialog holds two tables, the editable shortcuts and the fixed ones.
  % One of them should be listing the label core's fixed shortcuts, their
  % descriptions in the first column.
  tables = findall(shortcutsFigure, 'Type', 'uitable') ;
  isFixedTable = ...
    arrayfun(@(table)(size(table.Data, 1) == size(fixedShortcuts, 1) && ...
                      isequal(table.Data(:, 1), fixedShortcuts(:, 1))), ...
             tables) ;
  assert(any(isFixedTable), ...
         'The Shortcuts dialog does not show the label core''s fixed shortcuts.') ;
end  % function
