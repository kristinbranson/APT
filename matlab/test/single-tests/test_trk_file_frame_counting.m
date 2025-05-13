function test_trk_file_frame_counting()
  % Test counting of frames in .trk files
  [~, unittest_dir_path, ~] = get_test_project_paths() ;
  trk_folder_path = fullfile(unittest_dir_path,'trks') ;
  name_from_file_index = { 'carmen-underfly-gt-1.trk', ...
                           'from-ma-with-trx-for-all-targets.trk.part', ...
                           'from-ma-with-trx-for-single-target.trk', ...
                           'multianimal-single-view-gt.trk', ...
                           'problem.trk', ...
                           'sam-two-view-view-0.trk', ...
                           'sam-two-view-view-1.trk' }' ;
  true_frame_count_from_file_index = num2cell([500 401 201 11 201 1001 1001]') ; 

  % Define a function the check the frame count for a single file
  function check_frame_count(file_name, true_frame_count)
    file_path = fullfile(trk_folder_path, file_name) ;
    frame_count = TrkFile.getNFramesTracked(file_path) ;
    if frame_count ~= true_frame_count ,
      error('For test trk file %s the returned frame count was %d, but the true frame_count is %d', file_name, frame_count, true_frame_count) ;
    end
  end  % function

  % Call the function on each <filename, frame_count> pair
  cellfun(@check_frame_count, name_from_file_index, true_frame_count_from_file_index) ;
end  % function
