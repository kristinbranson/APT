classdef OurPeepsTestCase < matlab.unittest.TestCase
    methods (TestMethodSetup)
        function setup(obj) %#ok<MANU>
        end
    end

    methods (TestMethodTeardown)
      function teardown(obj) %#ok<MANU>
      end
    end

%     methods (Static)
%       function result = getJRCAdditionalBsubArgs()
%         if strcmp(get_user_name(), 'taylora') ,
%           result = '-P scicompsoft' ;
%         else
%           result = '' ;
%         end
%       end  % function        
%     end  % methods (Static)
    
    methods (Test)        
      function aliceTest(obj)
        testObj = TestAPT('name','alice');
        testObj.test_full('nets',{'deeplabcut'}, ...
                          'setup_params',{'simpleprojload',1});  % took mdn out b/c deprecated and doesn't seem to work
        obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000) ;          
      end  % function

      function roianTest(obj)
        testObj = TestAPT('name','roianma');
        testObj.test_setup('simpleprojload',1);
        testObj.test_train('net_type',[],'params',-1,'niters',1000);
        obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000) ;          
      end  % function

%       function roianMAGroneTest(obj)
%         testObj = TestAPT('name','roianma');
%         jrcAdditionalBsubArgs = obj.getJRCAdditionalBsubArgs() ;
%         testObj.test_full('nets',{'magrone'}, ...
%                           'setup_params',{'simpleprojload',1, ...
%                                           'jrcgpuqueue','gpu_a100', ...
%                                           'jrcnslots',4, ...
%                                           'jrcAdditionalBsubArgs',jrcAdditionalBsubArgs}, ...
%                           'backend','bsub');
%            % empty nets means test all nets
%         obj.verifyTrue(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000) ;          
%       end  % function        
    end  % test methods
 end  % classdef
