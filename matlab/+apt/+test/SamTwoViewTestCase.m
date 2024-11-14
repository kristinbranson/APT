classdef SamTwoViewTestCase < matlab.unittest.TestCase
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
    function trainingAndTrackingTest(obj)
      testObj = TestAPT('name','sam2view');
      setup_params = apt.test.SamTwoViewTestCase.getSetupParams() ;
      testObj.test_setup(setup_params{:}) ;
      testObj.test_train('backend','bsub');
      obj.verifyEqual(testObj.labeler.tracker.algorithmName, 'mdn_joint_fpn', 'Training was not done with GRONe aka mdn_joint_fpn') ;
      did_train_enough = all(testObj.labeler.tracker.trnLastDMC.iterCurr>=1000) ;
      obj.verifyTrue(did_train_enough, 'Failed to complete all training iterations') ;
      obj.verifyEmpty(testObj.labeler.tracker.trkP, 'testObj.labeler.tracker.trkP is nonempty---it should be empty before tracking') ;
      testObj.test_track('backend','bsub');
      obj.verifyNotEmpty(testObj.labeler.tracker.trkP, 'testObj.labeler.tracker.trkP is empty---it should be nonempty after tracking') ;
      obj.verifyClass(testObj.labeler.tracker.trkP, 'TrkFile', 'testObj.labeler.tracker.trkP is not of class TrkFile after tracking') ;
      obj.verifyClass(testObj.labeler.tracker.trkP.pTrk, 'cell', 'testObj.labeler.tracker.trkP.pTrk is not of class cell after tracking') ;
      obj.verifyNotEmpty(testObj.labeler.tracker.trkP.pTrk, 'testObj.labeler.tracker.trkP.pTrk is empty---it should be nonempty after tracking') ;
      obj.verifySize(testObj.labeler.tracker.trkP.pTrk{1}, [14 2 201], 'After tracking, testObj.labeler.tracker.trkP.pTrk{10} is the wrong size') ;
      obj.verifyTrue(all(isfinite(testObj.labeler.tracker.trkP.pTrk{1}), 'all'), ...
                     'After tracking, testObj.labeler.tracker.trkP.pTrk{1} has non-finite elements') ;
    end  % function

%     function trackingTest(obj)
%       testObj = TestAPT('name','argrone');
%       setup_params = apt.test.ARGRONeTestCase.getSetupParams() ;
%       testObj.test_setup(setup_params{:}) ;
%       testObj.test_track('backend','bsub');
%       obj.verifyNotEmpty(testObj.labeler.tracker.trkP, 'testObj.labeler.tracker.trkP is empty---it should be nonempty after tracking') ;
%       obj.verifyClass(testObj.labeler.tracker.trkP, 'TrkFile', 'testObj.labeler.tracker.trkP is not of class TrkFile after tracking') ;
%       obj.verifyClass(testObj.labeler.tracker.trkP.pTrk, 'cell', 'testObj.labeler.tracker.trkP.pTrk is not of class cell after tracking') ;
%       obj.verifyNotEmpty(testObj.labeler.tracker.trkP.pTrk, 'testObj.labeler.tracker.trkP.pTrk is empty---it should be nonempty after tracking') ;
%       obj.verifySize(testObj.labeler.tracker.trkP.pTrk{1}, [10 2 101], 'After tracking, testObj.labeler.tracker.trkP.pTrk{1} is the wrong size') ;
%       obj.verifyTrue(all(isfinite(testObj.labeler.tracker.trkP.pTrk{1}), 'all'), ...
%                      'After tracking, testObj.labeler.tracker.trkP.pTrk{1} has non-finite elements') ;
%     end  % function
    
%     function groundTruthTest(obj)
%       testObj = TestAPT('name','argrone');
%       setup_params = apt.test.ARGRONeTestCase.getSetupParams() ;
%       testObj.test_setup(setup_params{:}) ;
%       testObj.test_gtcompute('backend','bsub') ;
%       tbl = testObj.labeler.gtTblRes ;
%       obj.verifyTrue(isequal(size(tbl), [1539 11]), ...
%                      'After GT tracking, testObj.labeler.gtTblRes is the wrong size') ;      
%       err = tbl.meanL2err ;
%       obj.verifyLessThan(median(err, 'omitnan'), 10, 'Median value of testObj.labeler.gtTblRes.meanL2err is too large') ;
%     end  % function
    
  end  % methods (Test)
end  % classdef
