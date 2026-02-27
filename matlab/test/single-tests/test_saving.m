function test_saving()
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl' ;
  if ispc()
    project_file_path = strrep(linux_project_file_path, '/groups/branson/bransonlab', 'Z:') ;
    replace_path = { '/groups/branson/bransonlab', 'Z:' } ;
  else
    project_file_path = linux_project_file_path ;
    replace_path = [] ;
  end

  % Start APT, load project, do save-as, then close APT
  temp_file_path = load_project_and_save_as(project_file_path, replace_path) ;
  cleaner3 = onCleanup(@()(delete(temp_file_path))) ;

  % Now load the copy that was saved
  [labeler, controller] = ...
    StartAPT('projfile', temp_file_path, ...
             'replace_path', replace_path, ...
             'isInDebugMode', true, ...
             'isInYodaMode', true) ;
  cleaner = onCleanup(@()(delete(controller))) ;  % this will delete labeler too
  cleaner2 = onCleanup(@()(delete(labeler))) ;  % but just to be sure

  % Do verification
  % For now we'll have the test pass as long as saving and loading works without
  % erroring.
end  % function    



function temp_file_path = load_project_and_save_as(project_file_path, replace_path)
  [labeler, controller] = ...
    StartAPT('projfile', project_file_path, ...
             'replace_path', replace_path, ...
             'isInDebugMode', true, ...
             'isInYodaMode', true) ;
  cleaner = onCleanup(@()(delete(controller))) ;  % this will delete labeler too
  cleaner2 = onCleanup(@()(delete(labeler))) ;  % but just to be sure

  % Do Save As...
  temp_file_path = tempname() ;
  labeler.projSave(temp_file_path) ;
end
