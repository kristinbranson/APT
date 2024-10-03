% % Simplest way to test:
% testObj = TestAPT('name','alice'); 
% testObj.test_full('nets',{'deeplabcut'});  % took mdn out b/c deprecated and doesn't seem to work
% testObj = [] ;

% % MA/roian
% testObj = TestAPT('name','roianma');
% testObj.test_setup('simpleprojload',1);
% testObj.test_train('net_type',[],'params',-1,'niters',1000);  
% testObj = [] ;

% MA/roian, Kristin's suggestion:
testObj = TestAPT('name','roianma');  
testObj.test_full('nets',{},'setup_params',{'simpleprojload',1,'jrcgpuqueue','gpu_a100','jrcnslots',4},'backend','bsub');
% testObj.test_full('nets',{'Stg1tddobj_Stg2tdpobj','magrone','Stg1tddht_Stg2tdpht','maopenpose'},...
%                   'setup_params',{'simpleprojload',1,'jrcgpuqueue','gpu_rtx8000','jrcnslots',4},...
%                   'backend','docker');
  % empty nets means test all nets
% testObj = [] ;

% % Carmen/GT workflow (proj on JRC/dm11)
% testObj = TestAPT('name','carmen');
% testObj.test_setup('simpleprojload',1);
% testObj.test_train('backend','bsub');
% testObj.test_track('backend','bsub');
% testObj.test_gtcompute('backend','bsub');
