classdef CarmenTestCase < matlab.unittest.TestCase
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
      testObj = TestAPT('name','carmen');
      setup_params = apt.test.CarmenTestCase.getSetupParams() ;
      backend_params = apt.test.CarmenTestCase.getBackendParams() ;
      testObj.test_setup(setup_params{:}) ;
      backend = fif(ispc(), 'docker', 'bsub') ;      
      testObj.test_train('backend',backend, ...
                         'backend_params', backend_params);
      did_train_enough = (testObj.labeler.tracker.trnLastDMC.iterCurr>=1000) ;
      obj.verifyTrue(did_train_enough, 'Failed to complete all training iterations') ;
    end  % function

    function trackingTest(obj)
      testObj = TestAPT('name','carmen_tracking');
      setup_params = apt.test.CarmenTestCase.getSetupParams() ;
      backend_params = apt.test.CarmenTestCase.getBackendParams() ;
      testObj.test_setup(setup_params{:}) ;
      backend = fif(ispc(), 'docker', 'bsub') ;      
      testObj.test_track('backend',backend, ...
                         'backend_params', backend_params);
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
      backend_params = apt.test.CarmenTestCase.getBackendParams() ;
      testObj.test_setup(setup_params{:}) ;
      backend = fif(ispc(), 'docker', 'bsub') ;            
      testObj.test_gtcompute('backend',backend, ...
                             'backend_params', backend_params);
      tbl = testObj.labeler.gtTblRes ;
      obj.verifyTrue(isequal(size(tbl), [1539 11]), ...
                     'After GT tracking, testObj.labeler.gtTblRes is the wrong size') ;      
      err = tbl.meanL2err ;
      obj.verifyLessThan(median(err, 'omitnan'), 10, 'Median value of testObj.labeler.gtTblRes.meanL2err is too large') ;
    end  % function
    
  end  % methods (Test)
end  % classdef
