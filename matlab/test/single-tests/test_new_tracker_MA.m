function test_new_tracker_MA()
  % Test creation of new trackers in an MA project.
  [~, unittest_dir_path, replace_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'four-points-testing-2025-04-11-with-rois-added-and-fewer-smaller-avi-movies.lbl');  

  % Launch APT
  [labeler, controller] = StartAPT();
  oc1 = onCleanup(@()(delete(controller)));  
  oc2 = onCleanup(@()(delete(labeler)));  

  % Set the labeler to silent mode for batch operation
  labeler.silent = true ;

  % Load the named project
  labeler.projLoadGUI(project_file_path, 'replace_path', replace_path);

  % Get the list of available tracker types
  [maposenets, mabboxnets, saposenets] = Labeler.getAllTrackerTypes();
  stage1NetTypes = vertcat(maposenets, mabboxnets);
  % stage1NetModes = vertcat(repmat(DLNetMode.multiAnimalTDDetectHT, size(maposenets)), ...
  %                          repmat(DLNetMode.multiAnimalTDDetectObj, size(mabboxnets)));
  stage2NetTypes = saposenets;  
  % stage2NetModes = vertcat(repmat(DLNetMode.multiAnimalTDPoseHT, size(maposenets)), ...
  %                          repmat(DLNetMode.multiAnimalTDPoseObj, size(mabboxnets)));

  stage1NetTypeCount = numel(stage1NetTypes);
  stage2NetTypeCount = numel(stage2NetTypes);
  totalTrackerCount =  stage1NetTypeCount * stage2NetTypeCount ;
  totalTrackersCreatedCount = 0 ;
  for i = 1 : stage1NetTypeCount
    desiredStage1NetType = stage1NetTypes(i);
    for j = 1 : stage2NetTypeCount
      desiredStage2NetType = stage2NetTypes(j);      
      labeler.trackMakeNewTrackerGivenNetTypes([desiredStage1NetType desiredStage2NetType]);
      pause(0.1);
      stage1NetType = labeler.tracker.stage1Tracker.trnNetType ;
      stage2NetType = labeler.tracker.trnNetType ;
      if ~( stage1NetType == desiredStage1NetType && stage2NetType == desiredStage2NetType )
        error('Failed to create new tracker of type %s+%s', char(desiredStage1NetType), char(desiredStage2NetType)) ;
      end
      totalTrackersCreatedCount = totalTrackersCreatedCount + 1 ;
    end
  end

  % If get here then all is well
  fprintf('%d of %d MA tracker types created.\n', totalTrackersCreatedCount, totalTrackerCount) ;
end  % function
