classdef AliceTestCase < matlab.unittest.TestCase
    methods (TestMethodSetup)
        function setup(obj) %#ok<MANU>
        end
    end

    methods (TestMethodTeardown)
      function teardown(obj) %#ok<MANU>
      end
    end
    
    methods (Test)        
      function aliceTest(obj)
        testObj = TestAPT('name','alice');
        testObj.test_full('nets',{'deeplabcut'}, ...
                          'setup_params',{'simpleprojload',1});  % took mdn out b/c deprecated and doesn't seem to work
        obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000, 'Failed to complete all training iterations') ;          
      end  % function
    end  % test methods
 end  % classdef
