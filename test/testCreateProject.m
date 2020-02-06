addpath('../matlab');
APT.setpath;
ref_lbl = '/work/mayank/work/FlySpaceTime/multitarget_bubble_expandedbehavior_20180425_fixederrors_fixed.lbl';
nviews = 1;
npts = 17;
has_trx = true;
proj_name = 'alice_test';
sz = 90;

old = loadLbl(ref_lbl);

%% Create the new project
lObj = Labeler;
cfg = Labeler.cfgGetLastProjectConfigNoView;
cfg.NumViews = nviews;
cfg.NumLabelPoints = npts;
cfg.Trx.HasTrx = has_trx;
cfg.ViewNames = {};
cfg.LabelPointNames = {};
cfg.Track.Enable = true;
cfg.ProjectName = proj_name;
FIELDS2DOUBLIFY = {'Gamma' 'FigurePos' 'AxisLim' 'InvertMovie' 'AxFontSize' 'ShowAxTicks' 'ShowGrid'};
for i=1:numel(cfg.View)  
  cfg.View(i) = ProjectSetup('structLeavesStr2Double',cfg.View(i),FIELDS2DOUBLIFY);
end

lObj.initFromConfig(cfg);
lObj.projNew(cfg.ProjectName);

%% Add movies

old.movieFilesAll = FSPath.macroReplace(old.movieFilesAll,old.projMacros);
old.movieFilesAllGT = FSPath.macroReplace(old.movieFilesAllGT,old.projMacros);
if has_trx
    old.trxFilesAll = FSPath.macroReplace(old.trxFilesAll,old.projMacros);
    old.trxFilesAllGT = FSPath.macroReplace(old.trxFilesAllGT,old.projMacros);    
end

nmov = size(old.movieFilesAll,1);
for ndx = 1:nmov
    if has_trx
        lObj.movieAdd(old.movieFilesAll{ndx,1},old.trxFilesAll{ndx,1});
    else
        lobj.movieSetAdd(old.movieFilesAll(ndx,:));
    end
end

lObj.movieSet(1,'isFirstMovie',true);

%% add the labels

lc = lObj.lblCore;

for ndx = 1:nmov
    lObj.movieSet(ndx);
    old_labels = SparseLabelArray.full(old.labeledpos{ndx});
    ntgts = size(old_labels,4);
    for itgt = 1:ntgts
        
        frms = find(squeeze(any(any(~isnan(old_labels(:,:,:,itgt)),1),2)));
        if isempty(frms), continue; end        
        for fr = 1:numel(frms)
            cfr = frms(fr);
            lObj.setFrameAndTarget(cfr,itgt);
            for pt = 1:npts
                lc.hPts(pt).XData = old_labels(pt,1,cfr,itgt);
                lc.hPts(pt).YData = old_labels(pt,2,cfr,itgt);
            end
            lc.acceptLabels()
        end        
    end    
end

%% Start from label file

%lObj = Labeler;
%lObj.projLoad(ref_lbl);

%% Set the algorithm.

alg = 'mdn';
nalgs = numel(lObj.trackersAll);

tndx = 0;
for ix = 1:nalgs
    if strcmp(lObj.trackersAll{ix}.algorithmName,'mdn')
        tndx = ix;
    end
end

assert(tndx > 0)
lObj.trackSetCurrentTracker(tndx);

% set some params
lObj.trackParams.ROOT.DeepTrack.GradientDescent.dl_steps = 1000;
lObj.trackParams.ROOT.ImageProcessing.MultiTarget.TargetCrop.Radius = sz;
lObj.trackParams.ROOT.ImageProcessing.MultiTarget.TargetCrop.AlignUsingTrxTheta = has_trx;

%% Set the Backend

beType = DLBackEnd.Docker;
be = DLBackEndClass(beType,lObj.trackGetDLBackend);
lObj.trackSetDLBackend(be);


%% train

handles = lObj.gdata;
oc1 = onCleanup(@()ClearStatus(handles));
wbObj = WaitBarWithCancel('Training');
oc2 = onCleanup(@()delete(wbObj));
centerOnParentFigure(wbObj.hWB,handles.figure);
handles.labelerObj.trackRetrain('retrainArgs',{'wbObj',wbObj});
if wbObj.isCancel
  msg = wbObj.cancelMessage('Training canceled');
  msgbox(msg,'Train');
end

