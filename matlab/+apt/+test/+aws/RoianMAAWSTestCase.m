classdef RoianMAAWSTestCase < matlab.unittest.TestCase
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
         'jrcAdditionalBsubArgs',jrcAdditionalBsubArgs, ...
         'awsKeyName', 'alt_taylora-ws4', ...
         'awsPEM', '/home/taylora/.ssh/alt_taylora-ws4.pem', ...
         'awsInstanceID', 'i-0da079e9b4d2d66b9'} ;
    end  % function
  end  % methods (Static)
  
  methods (Test)
    function magroneAWSTest(obj)
      testObj = TestAPT('name','roianma2');
      setup_params = apt.test.aws.RoianMAAWSTestCase.getSetupParams() ;
      backend_params = apt.test.aws.RoianMAAWSTestCase.getBackendParams() ;
      niters = 1000 ;
      testObj.test_full('nets',{'magrone'}, ...
                        'setup_params',setup_params, ...
                        'backend','aws', ...
                        'backend_params', backend_params, ...
                        'niters',niters);
      obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=niters, 'Failed to complete all training iterations') ;
    end  % function

    % function groundTruthTest(obj)
    %   testObj = TestAPT('name','roianma2gt');
    %   setup_params = apt.test.aws.RoianMAAWSTestCase.getSetupParams() ;
    %   backend_params = apt.test.aws.RoianMAAWSTestCase.getBackendParams() ;
    %   testObj.test_setup(setup_params{:}) ;
    %   testObj.test_gtcompute('backend','aws', ...
    %                          'backend_params', backend_params) ;
    %   tbl = testObj.labeler.gtTblRes ;
    %   obj.verifyTrue(isequal(size(tbl), [11 11]), ...
    %                  'After GT tracking, testObj.labeler.gtTblRes is the wrong size') ;      
    %   err = tbl.meanL2err ;
    %   obj.verifyLessThan(median(err(:), 'omitnan'), 50, 'Median value of testObj.labeler.gtTblRes.meanL2err(:) is too large') ;
    % end  % function
    
  end  % methods (Test)
end  % classdef
