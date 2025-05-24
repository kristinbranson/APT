[labeler, controller] = StartAPT('projfile', '~/apt/kb-tracking-issue-with-hacked-project/DanionellaWild_Grone_restitched_v2_resave_modded_2_another_100_iters.lbl','isInDebugMode', true, 'isInYodaMode', true) ;
load('repro.mat', 'toTrackStruct') ;
toTrackStruct.f0s = 1 ;
toTrackStruct.f1s = 200 ;
labeler.trackBatch('toTrack', toTrackStruct) ;
