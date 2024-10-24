classdef RoianMAManyNetsTestCase < matlab.unittest.TestCase
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
    function magroneTest(obj)
      testObj = TestAPT('name','roianma');
      setup_params = apt.test.RoianMAManyNetsTestCase.getSetupParams() ;
      testObj.test_full('nets',{'magrone'}, ...
                        'setup_params',setup_params, ...
                        'backend','bsub');
      obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000, 'Failed to complete all training iterations') ;
    end  % function

    function maopenposeTest(obj)
      testObj = TestAPT('name','roianma');
      setup_params = apt.test.RoianMAManyNetsTestCase.getSetupParams() ;
      testObj.test_full('nets',{'maopenpose'}, ...
                        'setup_params',setup_params, ...
                        'backend','conda');
      obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000, 'Failed to complete all training iterations') ;
    end  % function
    
    function multiCiDTest(obj)
      testObj = TestAPT('name','roianma');
      setup_params = apt.test.RoianMAManyNetsTestCase.getSetupParams() ;
      testObj.test_full('nets',{'multi_cid'}, ...
                        'setup_params',setup_params, ...
                        'backend','bsub');
      obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000, 'Failed to complete all training iterations') ;
    end  % function
    
    function multiDekrTest(obj)
      testObj = TestAPT('name','roianmammpose1');
      setup_params = apt.test.RoianMAManyNetsTestCase.getSetupParams() ;
      testObj.test_full('nets',{'multi_dekr'}, ...
                        'setup_params',setup_params, ...
                        'backend','docker');
      obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000, 'Failed to complete all training iterations') ;
    end  % function
    
    function longName1Test(obj)
      testObj = TestAPT('name','roianma');
      setup_params = apt.test.RoianMAManyNetsTestCase.getSetupParams() ;
      testObj.test_full('nets',{'ma_top_down_ht_tddht_tdpht'}, ...
                        'setup_params',setup_params, ...
                        'backend','bsub');
      obj.verifyTrue(all(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000), 'Failed to complete all training iterations') ;
    end  % function
    
    function longName2Test(obj)
      testObj = TestAPT('name','roianma');
      setup_params = apt.test.RoianMAManyNetsTestCase.getSetupParams() ;
      testObj.test_full('nets',{'ma_top_down_bbox_tddobj_tdpobj'}, ...
                        'setup_params',setup_params, ...
                        'backend','bsub');
      obj.verifyTrue(all(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000), 'Failed to complete all training iterations') ;
    end  % function
    
    function longName3Test(obj)
      testObj = TestAPT('name','roianma');
      setup_params = apt.test.RoianMAManyNetsTestCase.getSetupParams() ;
      testObj.test_full('nets',{'ma_top_down_custom_ht_tddht_tdpht'}, ...
                        'setup_params',setup_params, ...
                        'backend','bsub');
      obj.verifyTrue(all(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000), 'Failed to complete all training iterations') ;
    end  % function
    
    function longName4Test(obj)
      testObj = TestAPT('name','roianma');
      setup_params = apt.test.RoianMAManyNetsTestCase.getSetupParams() ;
      testObj.test_full('nets',{'ma_top_down_custom_bbox_tddobj_tdpobj'}, ...
                        'setup_params',setup_params, ...
                        'backend','bsub');
      obj.verifyTrue(all(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000), 'Failed to complete all training iterations') ;
    end  % function
    
  end  % methods (Test)
end  % classdef