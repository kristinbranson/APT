function test_new_tracker_SA()
  % Test creation of new trackers in an SA project.
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'multitarget_bubble_training_20210523_allGT_AR_MAAPT_grone2_UT_resaved_3.lbl');  

  % Launch APT
  [labeler, controller] = StartAPT();
  oc1 = onCleanup(@()(delete(controller)));  
  oc2 = onCleanup(@()(delete(labeler)));  

  % Set the labeler to silent mode for batch operation
  labeler.silent = true ;

  % Load the named project
  labeler.projLoadGUI(project_file_path, 'replace_path', replace_path);

  % Get the list of available tracker types
  [~, ~, saposenets] = Labeler.getAllTrackerTypes();
  netTypeCount = numel(saposenets);
  totalTrackerCount =  netTypeCount ;
  totalTrackersCreatedCount = 0 ;  
  for i = 1 : netTypeCount
    desiredNetType = saposenets(i);
    labeler.trackMakeNewTrackerGivenNetTypes(desiredNetType);
    totalTrackersCreatedCount = totalTrackersCreatedCount + 1 ;
    pause(0.1);
    netType = labeler.tracker.trnNetType ;
    if ~( netType == desiredNetType )
      error('Failed to create new tracker of type %s', char(desiredSNetType)) ;
    end
  end

  % If get here then all is well
  fprintf('%d of %d SA tracker types created.\n', totalTrackersCreatedCount, totalTrackerCount) ;
end  % function
