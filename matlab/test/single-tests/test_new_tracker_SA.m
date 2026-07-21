function test_new_tracker_SA()
  % Test creation of new trackers in an SA project.
  linux_project_file_path = '/groups/branson/bransonlab/apt/unittest/multitarget_bubble_training_20210523_allGT_AR_MAAPT_grone2_UT_resaved_3.lbl' ;
  [project_file_path, replace_path] = localize_test_project_path(linux_project_file_path) ;

  % Launch APT
  [labeler, controller] = StartAPT();
  oc1 = onCleanup(@()(delete(controller)));  
  oc2 = onCleanup(@()(delete(labeler)));  

  % Put the labeler in batch mode so prompts use defaults
  labeler.isInBatchMode = true ;

  % Load the named project
  labeler.projLoad(project_file_path, 'replace_path', replace_path) ;

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
