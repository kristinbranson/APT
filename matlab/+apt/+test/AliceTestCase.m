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
        % Modify the training parameters so that the set max iters is honored by the
        % DeepLabCut model        
        training_params = struct('dlc_override_dlsteps', {true}) ;  % scalar struct
        niters = 1000 ;
        testObj = TestAPT('name','alice');
        testObj.test_full('nets',{'deeplabcut'}, ...
                          'params', training_params, ...
                          'niters',niters, ...
                          'setup_params',{'simpleprojload',1});  % took mdn out b/c deprecated and doesn't seem to work
        obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=niters, 'Failed to complete all training iterations') ;          
      end  % function
    end  % test methods
 end  % classdef
