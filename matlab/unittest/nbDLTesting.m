%%
LBL = '/groups/branson/bransonlab/apt/test/testproj_flybub/proj_nomacros.lbl'; 

%% Proj load
lObj = StartAPT;
lObj.projLoadGUI(LBL);

%% Set tracker
lObj.trackersAll % all tracker objects
lObj.trackSetCurrentTracker(2); % MDN
t = lObj.tracker; % DeepTracker/MDN tracking object

%% Parameters
sPrm = lObj.trackGetTrainingParams() % All APT parameters
% ... modify sPrm ...
lObj.trackSetTrainingParams(sPrm);

%% Backend
be = DLBackEnd.Bsub; % AWS, Bsub, Docker
lObj.trackSetDLBackend(be);

% (for aws, need a couple additional steps to create/configure AWSec2
% instance)

%% Train
%wbObj = WaitBarWithCancel('Training');
lObj.train();

%% State of Train
t.trnPrintLogs
t.trnPrintModelChainDir
t.trnKill
t.bgTrnReset % occassionally necessary to manually reset training monitor 

%% Track
mfts = MFTSetEnum.CurrMovTgtEveryFrame; % tons of options avail in enumeration; also, custom thing possible 
wbObj = WaitBarWithCancel('Tracking');
lObj.track(mfts,'wbObj',wbObj);

%% Tracking results
mIdx = MovieIndex(1:lObj.nmovies);
[trkfileObjs,tfHasRes] = t.getTrackingResults(mIdx);

%% Proj save
lObj.projSaveRaw('/path/to/lblfile/to/save.lbl');



