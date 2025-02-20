classdef RoianMATestCase < matlab.unittest.TestCase
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
    function defaultTrainingTest(obj)
      backend_params = apt.test.RoianMATestCase.getBackendParams() ;
      testObj = TestAPT('name','roianma2');
      testObj.test_setup('simpleprojload',1);
      testObj.test_train('net_type',[],'params',-1,'niters',1000, ...
                         'backend_params', backend_params) ;
      obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000, 'Failed to complete all training iterations') ;
    end  % function

    function magroneTest(obj)
      testObj = TestAPT('name','roianma2');
      setup_params = apt.test.RoianMATestCase.getSetupParams() ;
      backend_params = apt.test.CarmenTestCase.getBackendParams() ;
      testObj.test_full('nets',{'magrone'}, ...
                        'setup_params',setup_params, ...
                        'backend',fif(ispc(), 'docker', 'conda'), ...
                        'backend_params', backend_params) ;
      obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000, 'Failed to complete all training iterations') ;
    end  % function

    function maopenposeTest(obj)
      testObj = TestAPT('name','roianma2');
      setup_params = apt.test.RoianMATestCase.getSetupParams() ;
      backend_params = apt.test.CarmenTestCase.getBackendParams() ;
      testObj.test_full('nets',{'maopenpose'}, ...
                        'setup_params',setup_params, ...
                        'backend',fif(ispc(), 'docker', 'bsub'), ...
                        'backend_params', backend_params) ;
      obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000, 'Failed to complete all training iterations') ;
    end  % function
    
    function multiCiDTest(obj)
      testObj = TestAPT('name','roianma2');
      setup_params = apt.test.RoianMATestCase.getSetupParams() ;
      backend_params = apt.test.CarmenTestCase.getBackendParams() ;
      testObj.test_full('nets',{'multi_cid'}, ...
                        'setup_params',setup_params, ...
                        'backend',fif(ispc(), 'docker', 'bsub'), ...
                        'backend_params', backend_params) ;
      
      obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000, 'Failed to complete all training iterations') ;
    end  % function
    
    function multiDekrTest(obj)
      testObj = TestAPT('name','roianma2mmpose1');
      setup_params = apt.test.RoianMATestCase.getSetupParams() ;
      backend_params = apt.test.CarmenTestCase.getBackendParams() ;
      testObj.test_full('nets',{'multi_dekr'}, ...
                        'setup_params',setup_params, ...
                        'backend','docker', ...
                        'backend_params', backend_params) ;
      obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000, 'Failed to complete all training iterations') ;
    end  % function
    
    function longName1Test(obj)
      testObj = TestAPT('name','roianma2');
      setup_params = apt.test.RoianMATestCase.getSetupParams() ;
      backend_params = apt.test.CarmenTestCase.getBackendParams() ;
      testObj.test_full('nets',{'ma_top_down_ht_tddht_tdpht'}, ...
                        'setup_params',setup_params, ...
                        'backend',fif(ispc(), 'docker', 'bsub'), ...
                        'backend_params', backend_params) ;
      obj.verifyTrue(all(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000), 'Failed to complete all training iterations') ;
    end  % function
    
    function longName2Test(obj)
      testObj = TestAPT('name','roianma2');
      setup_params = apt.test.RoianMATestCase.getSetupParams() ;
      backend_params = apt.test.CarmenTestCase.getBackendParams() ;
      testObj.test_full('nets',{'ma_top_down_bbox_tddobj_tdpobj'}, ...
                        'setup_params',setup_params, ...
                        'backend',fif(ispc(), 'docker', 'bsub'), ...
                        'backend_params', backend_params) ;
      obj.verifyTrue(all(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000), 'Failed to complete all training iterations') ;
    end  % function
    
    function longName3Test(obj)
      testObj = TestAPT('name','roianma2');
      setup_params = apt.test.RoianMATestCase.getSetupParams() ;
      backend_params = apt.test.CarmenTestCase.getBackendParams() ;
      testObj.test_full('nets',{'ma_top_down_custom_ht_tddht_tdpht'}, ...
                        'setup_params',setup_params, ...
                        'backend',fif(ispc(), 'docker', 'bsub'), ...
                        'backend_params', backend_params) ;
      obj.verifyTrue(all(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000), 'Failed to complete all training iterations') ;
    end  % function
    
    function longName4Test(obj)
      testObj = TestAPT('name','roianma2');
      setup_params = apt.test.RoianMATestCase.getSetupParams() ;
      backend_params = apt.test.CarmenTestCase.getBackendParams() ;
      testObj.test_full('nets',{'ma_top_down_custom_bbox_tddobj_tdpobj'}, ...
                        'setup_params',setup_params, ...
                        'backend',fif(ispc(), 'docker', 'bsub'), ...
                        'backend_params', backend_params) ;
      obj.verifyTrue(all(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000), 'Failed to complete all training iterations') ;
    end  % function
    
    function groundTruthTest(obj)
      testObj = TestAPT('name','roianma2gt');
      setup_params = apt.test.RoianMATestCase.getSetupParams() ;
      backend_params = apt.test.CarmenTestCase.getBackendParams() ;
      testObj.test_setup(setup_params{:}) ;
      testObj.test_gtcompute('backend',fif(ispc(), 'docker', 'bsub'), ...
                             'backend_params', backend_params) ;
      tbl = testObj.labeler.gtTblRes ;
      obj.verifyTrue(isequal(size(tbl), [11 11]), ...
                     'After GT tracking, testObj.labeler.gtTblRes is the wrong size') ;      
      err = tbl.meanL2err ;
      obj.verifyLessThan(median(err(:), 'omitnan'), 50, 'Median value of testObj.labeler.gtTblRes.meanL2err(:) is too large') ;
    end  % function
    
  end  % methods (Test)
end  % classdef
