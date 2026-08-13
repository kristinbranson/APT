function test_label_outlier_gui_smoke()
  % Headless smoke test for the de-guidata'd "Suspicious Labels" dialog.
  %
  % label_outlier_gui.m (a bare uifigure function with state in guidata) is now
  % the subcontroller class LabelOutlierController.  The outlier computation
  % lives in the static fromLabeler() factory; the constructor builds the view
  % from already-computed error tables.  This test builds the view directly
  % from fake error tables (so it needs no real project) and asserts the figure
  % opens with the expected widgets.

  errTables = {makeFakeErrorTable_(), makeFakeErrorTable_(), makeFakeErrorTable_()} ;
  labeler = struct() ;  % only used by the cell-click callback, which we do not fire
  controller = LabelOutlierController(labeler, errTables) ;
  cleaner = onCleanup(@()(delete(controller))) ;

  assert(~isempty(controller.hFig) && isgraphics(controller.hFig, 'figure'), ...
         'LabelOutlierController did not create a valid figure') ;
  assert(strcmp(controller.hFig.Name, 'Suspicious Labels'), ...
         'Dialog figure has unexpected Name "%s"', controller.hFig.Name) ;
  assert(isgraphics(controller.buttonGroup_) && isgraphics(controller.table_), ...
         'Button group / table were not created') ;

  radiobuttons = findall(controller.hFig, 'Type', 'uiradiobutton') ;
  assert(numel(radiobuttons) == 3, ...
         'Expected 3 outlier-type radiobuttons, found %d', numel(radiobuttons)) ;

  fprintf('test_label_outlier_gui_smoke: PASSED\n') ;
end  % function

function t = makeFakeErrorTable_()
  % A 5-column table shaped like label_outliers' output (Mov/Frm/Lbl plus two
  % score columns), matching the dialog's 5-entry ColumnWidth.
  Mov = [1 ; 2 ; 3] ;
  Frm = [10 ; 20 ; 30] ;
  Lbl = [1 ; 1 ; 2] ;
  Err = [0.5 ; 0.3 ; 0.1] ;
  Pt = [4 ; 5 ; 6] ;
  t = table(Mov, Frm, Lbl, Err, Pt) ;
end  % function
