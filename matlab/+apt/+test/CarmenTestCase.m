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
      obj.verifyTrue(did_train_enough) ;
      if ~did_train_enough ,
        % Don't both with further stuff if training didn't work
        return
      end
      % Maybe these things below should be their own tests?      
      testObj.test_track('backend','bsub');
      testObj.test_gtcompute('backend','bsub');
    end  % function
  end  % methods (Test)
end  % classdef
