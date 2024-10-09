classdef OurPeepsTestCase < matlab.unittest.TestCase
    methods (TestMethodSetup)
        function setup(self) %#ok<MANU>
        end
    end

    methods (TestMethodTeardown)
        function teardown(self) %#ok<MANU>
        end
    end

    methods (Test)        
        function aliceTest(obj)
          testObj = TestAPT('name','alice');
          testObj.test_full('nets',{'deeplabcut'});  % took mdn out b/c deprecated and doesn't seem to work
          obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000) ;          
        end        

        function roianTest(obj)
          testObj = TestAPT('name','roianma');
          testObj.test_setup('simpleprojload',1);
          testObj.test_train('net_type',[],'params',-1,'niters',1000);
          obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000) ;          
        end        
    end  % test methods
 end  % classdef
