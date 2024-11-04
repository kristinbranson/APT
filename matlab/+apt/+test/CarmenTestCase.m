classdef CarmenTestCase < matlab.unittest.TestCase
  methods (Static)
    function result = getSetupParams()
      if strcmp(get_user_name(), 'taylora') ,
        jrcAdditionalBsubArgs = '-P scicompsoft' ;
      else
        jrcAdditionalBsubArgs = '' ;
      end
      result = ...
        {'simpleprojload',1, ...
         'jrcgpuqueue','gpu_a100', ...
         'jrcnslots',4, ...
         'jrcAdditionalBsubArgs',jrcAdditionalBsubArgs} ;
    end  % function
  end  % methods (Static)
  
  methods (Test)
    function trainingTest(obj)
      testObj = TestAPT('name','carmen');
      setup_params = apt.test.CarmenTestCase.getSetupParams() ;
      testObj.test_setup(setup_params{:}) ;
      testObj.test_train('backend','bsub');
      did_train_enough = (testObj.labeler.tracker.trnLastDMC.iterCurr>=1000) ;
      obj.verifyTrue(did_train_enough, 'Failed to complete all training iterations') ;
%       if ~did_train_enough ,
%         % Don't both with further stuff if training didn't work
%         return
%       end
%       % Maybe these things below should be their own tests?      
%       testObj.test_track('backend','bsub');
%       testObj.test_gtcompute('backend','bsub');
    end  % function

    function trackingTest(obj)
      testObj = TestAPT('name','carmen_tracking');
      setup_params = apt.test.CarmenTestCase.getSetupParams() ;
      testObj.test_setup(setup_params{:}) ;
      testObj.test_track('backend','bsub');
      obj.verifyNotEmpty(testObj.labeler.tracker.trkP, 'testObj.labeler.tracker.trkP is empty---it should be nonempty after tracking') ;
      obj.verifyClass(testObj.labeler.tracker.trkP, 'TrkFile', 'testObj.labeler.tracker.trkP is not of class TrkFile after tracking') ;
      obj.verifyClass(testObj.labeler.tracker.trkP.pTrk, 'cell', 'testObj.labeler.tracker.trkP.pTrk is not of class cell after tracking') ;
      obj.verifyNotEmpty(testObj.labeler.tracker.trkP.pTrk, 'testObj.labeler.tracker.trkP.pTrk is empty---it should be nonempty after tracking') ;
      obj.verifySize(testObj.labeler.tracker.trkP.pTrk{1}, [10 2 101], 'After tracking, testObj.labeler.tracker.trkP.pTrk{1} is the wrong size') ;
      obj.verifyTrue(all(isfinite(testObj.labeler.tracker.trkP.pTrk{1}), 'all'), ...
                     'After tracking, testObj.labeler.tracker.trkP.pTrk{1} has non-finite elements') ;
    end  % function
    
    function groundTruthTest(obj)
      testObj = TestAPT('name','carmen_tracking');
      setup_params = apt.test.CarmenTestCase.getSetupParams() ;
      testObj.test_setup(setup_params{:}) ;
      testObj.test_track('backend','bsub');
      obj.verifyNotEmpty(testObj.labeler.tracker.trkP, 'testObj.labeler.tracker.trkP is empty---it should be nonempty after tracking') ;
      obj.verifyClass(testObj.labeler.tracker.trkP, 'TrkFile', 'testObj.labeler.tracker.trkP is not of class TrkFile after tracking') ;
      obj.verifyClass(testObj.labeler.tracker.trkP.pTrk, 'cell', 'testObj.labeler.tracker.trkP.pTrk is not of class cell after tracking') ;
      obj.verifyNotEmpty(testObj.labeler.tracker.trkP.pTrk, 'testObj.labeler.tracker.trkP.pTrk is empty---it should be nonempty after tracking') ;
      obj.verifySize(testObj.labeler.tracker.trkP.pTrk{1}, [10 2 101], 'After tracking, testObj.labeler.tracker.trkP.pTrk{1} is the wrong size') ;
      obj.verifyTrue(all(isfinite(testObj.labeler.tracker.trkP.pTrk{1}), 'all'), ...
                     'After tracking, testObj.labeler.tracker.trkP.pTrk{1} has non-finite elements') ;
      testObj.test_gtcompute('backend','bsub') ;
    end  % function
    
  end  % methods (Test)
end  % classdef
