function test_default_filename_export_gt_results()
  % The default file name for exporting GT results must be
  % <projdir>/<projfile>_gtresults.mat for a project loaded from a file.
  %
  % Regression test: Labeler.getDefaultFilenameExportGTResults() ran a second
  % macro replacement with an sMacro variable it never assigned, so
  % Evaluate > Export GT Results errored before its file dialog opened.

  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;

  [labeler, controller] = ...
    StartAPT('projfile', project_file_path, ...
             'replace_path', replace_path) ;
  cleaner = onCleanup(@()(delete(controller))) ;

  fileName = labeler.getDefaultFilenameExportGTResults() ;
  [projectDir, projectStem] = fileparts(labeler.projectfile) ;
  expectedFileName = linux_fullfile(projectDir, [projectStem '_gtresults.mat']) ;
  assert(ischar(fileName) && strcmp(fileName, expectedFileName), ...
         'Expected default GT results file name %s, got %s', expectedFileName, fileName) ;
end  % function
