%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  RCPR code                
% Copyright 2013 X.P. Burgos-Artizzu, P.Perona and Piotr Dollar.  
%  [xpburgos-at-gmail-dot-com]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]
%
%  Please cite our paper if you use the code:
%  Robust face landmark estimation under occlusion, 
%  X.P. Burgos-Artizzu, P. Perona, P. Dollar (c)
%  ICCV'13, Sydney, Australia
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PRE-REQUISITES
%Download, compile and install on path Piotr's Matlab Toolbox
%http://vision.ucsd.edu/~pdollar/toolbox/doc/
%
%% TO COMPILE mex files (private folder)
% Simply run rcprCompile.m 
%
%% DEMO EXAMPLES
% demoRCPR         - TEST pre-trained RCPR algorithm on COFW dataset
% FULL_demoRCPR.m  - RETRAIN and TEST RCPR algorithm on COFW dataset
% video_tracking/faceTracking.m   - RUN RCPR on a seq video from scratch
%                (make sure shapeGt functions are added to your path)
%% POSE FUNCTIONS
% ---all other cpr/rcpr functions ---
% shapeGt.m        - utils for handling landmarks 
% rcprTrain.m      - training RCPR code
% rcprTest.m       - testing RCPR code from several restarts
% rcprTest1.m	   - (auxiliary) testing RCPR code
% regTrain.m       - train a boosted regressor
% regApply.m       - apply a boosted regressor
% selectCorrFeat.m - select features based on correlation