classdef SamTwoViewTestCase < matlab.unittest.TestCase
  methods (Static)
    function result = getSetupParams()
      result = ...
        {'simpleprojload',1} ;
    end  % function
    function result = getBackendParams()
      if strcmp(get_user_name(), 'taylora') ,
        jrcAdditionalBsubArgs = '-P scicompsoft' ;
      else
        jrcAdditionalBsubArgs = '' ;
      end
      result = ...
        {'jrcgpuqueue','gpu_a100', ...
         'jrcnslots',4, ...
         'jrcAdditionalBsubArgs',jrcAdditionalBsubArgs} ;
    end  % function        
  end  % methods (Static)
  
  methods (Test)
    function trainingTest(obj)
      testObj = TestAPT('name','sam2view_training');
      setup_params = apt.test.SamTwoViewTestCase.getSetupParams() ;
      backend_params = apt.test.CarmenTestCase.getBackendParams() ;
      testObj.test_setup(setup_params{:}) ;
      iterationCount = 1000 ;
      testObj.test_train('backend',fif(ispc(), 'docker', 'bsub'), ...
                         'backend_params', backend_params, ...
                         'niters', iterationCount) ;      
      obj.verifyEqual(testObj.labeler.tracker.algorithmName, 'mdn_joint_fpn', 'Training was not done with GRONe aka mdn_joint_fpn') ;
      did_train_enough = all(testObj.labeler.tracker.trnLastDMC.iterCurr>=iterationCount) ;
      obj.verifyTrue(did_train_enough, 'Failed to complete all training iterations') ;
    end  % function

    function trackingTest(obj)
      testObj = TestAPT('name','sam2view_tracking');
      setup_params = apt.test.SamTwoViewTestCase.getSetupParams() ;
      backend_params = apt.test.CarmenTestCase.getBackendParams() ;
      testObj.test_setup(setup_params{:}) ;
      obj.verifyEmpty(testObj.labeler.tracker.trkP, 'testObj.labeler.tracker.trkP is nonempty---it should be empty before tracking') ;
      testObj.test_track('backend',fif(ispc(), 'docker', 'bsub'), ...
                         'backend_params', backend_params);
      obj.verifyNotEmpty(testObj.labeler.tracker.trkP, 'testObj.labeler.tracker.trkP is empty---it should be nonempty after tracking') ;
      obj.verifyClass(testObj.labeler.tracker.trkP, 'TrkFile', 'testObj.labeler.tracker.trkP is not of class TrkFile after tracking') ;
      obj.verifyClass(testObj.labeler.tracker.trkP.pTrk, 'cell', 'testObj.labeler.tracker.trkP.pTrk is not of class cell after tracking') ;
      obj.verifyNotEmpty(testObj.labeler.tracker.trkP.pTrk, 'testObj.labeler.tracker.trkP.pTrk is empty---it should be nonempty after tracking') ;
      obj.verifySize(testObj.labeler.tracker.trkP.pTrk{1}, [14 2 201], 'After tracking, testObj.labeler.tracker.trkP.pTrk{10} is the wrong size') ;
      obj.verifyTrue(all(isfinite(testObj.labeler.tracker.trkP.pTrk{1}), 'all'), ...
                     'After tracking, testObj.labeler.tracker.trkP.pTrk{1} has non-finite elements') ;
    end  % function

  end  % methods (Test)
end  % classdef
