function test_alice_project(obj)
  % Modify the training parameters so that the set max iters is honored by the
  % DeepLabCut model        

  [bransonlab_path, unittest_dir_path] = get_test_project_paths() ;
  project_file_path = fullfile(unittest_dir_path, 'alice/multitarget_bubble_expandedbehavior_20180425_allGT_MK_MDN04182019_updated_20250219.lbl') ;

  training_params = struct('dlc_override_dlsteps', {true}) ;  % scalar struct
  niters = 1000 ;
  testObj = TestAPT('name','alice');
  testObj.test_full('nets',{'deeplabcut'}, ...
                    'params', training_params, ...
                    'niters',niters, ...
                    'setup_params',{'simpleprojload',1});  % took mdn out b/c deprecated and doesn't seem to work
  obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=niters, 'Failed to complete all training iterations') ;          
end  % function
