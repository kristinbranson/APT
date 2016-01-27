function trainAL(tdfile,tpfile,varargin)
% Take a TrainData, TrainDataI, and TrainParams and produce a TrainRes

RCPR = '/groups/flyprojects/home/leea30/git/cpr';
JCTRAX = '/groups/flyprojects/home/leea30/git/jctrax';
PIOTR = '/groups/flyprojects/home/leea30/git/piotr_toolbox';

addpath(RCPR);
addpath(fullfile(RCPR,'video_tracking'));
addpath(fullfile(RCPR,'misc'));
addpath(fullfile(JCTRAX,'misc'));
addpath(fullfile(JCTRAX,'filehandling'));
addpath(genpath(PIOTR));

[rootdir,tdIfile,tdIfileVar] = myparse(varargin,...
    'rootdir','/groups/flyprojects/home/leea30/cpr/jan',... % place to look for files
    'tdIfile','',... % traindata Index file; if not specified, use td.iTrn
    'tdIfileVar','');

tdfilefull = fullfile(rootdir,tdfile);
td = load(tdfilefull);
td = td.td;
fprintf(1,'Loaded TD: %s\n',tdfilefull);
if ~isempty(tdIfile)
    tdIfilefull = fullfile(rootdir,tdIfile);
    tdI = load(tdIfilefull);
    tdI = tdI.(tdIfileVar);
    td.iTrn = tdI;
    
    fprintf(1,'tdIfile supplied: %s, var %s.\n',tdIfilefull,tdIfileVar);
else
    fprintf(1,'No tdIfile, using indices supplied with td.\n');
end
fprintf(1,'td.NTrn=%d\n',td.NTrn);

tpfilefull = fullfile(rootdir,tpfile);
tp = load(tpfilefull);
tp = tp.tp;
fprintf(1,'Using params file: %s\n',tpfilefull);
tpargs = tp.getPVs();

%% Train on training set
%cd(RCPR);
cmd = sprintf('git --git-dir=%s/.git rev-parse HEAD',RCPR);
[~,cmdout] = system(cmd);
sha = cmdout(1:5);

trname = FS.formTrainedClassifierName(tdfile,tdIfile,tdIfileVar,tpfile,sha);
trfilefull = fullfile(rootdir,trname);
fprintf('Training and saving results to: %s\n',trfilefull);

diary([trfilefull '.dry']);
[regModel,regPrm,prunePrm,phisPr,err] = train(...
    td.pGTTrn,td.bboxesTrn,td.ITrn,'savefile',trfilefull,tpargs{:});
diary off;


