classdef CPRLabelTracker < LabelTracker
  
  properties (Constant)
    TOKEN_SAVEPROPS = {'dataPPPrm' 'dataTS' ...
                       'trnRes' 'trnResTS' 'trnResPallMD' ...
                       'trkP' 'trkPFull' 'trkPTS' 'trkPMD'};
    TOKEN_LOADPROPS = {  ...
                       'trnRes' 'trnResTS' 'trnResPallMD' ...
                       'trkP' 'trkPFull' 'trkPTS' 'trkPMD'};
  end
  
  %% Data
  properties
    
    % Cached/working dataset. Contains all I/p/md for frames that have been
    % seen before by the tracker.
    % - Can be used for both training and tracking. 
    % - Current training frames stored in data.iTrn
    % - Current test/track frames stored in data.iTst
    % 
    % - All frames have an image, but need not have labels (p).
    % - If applicable, all frames are HE-ed the same way. 
    % - If applicable, all frames are PP-ed the same way.
    data 
    
    % Struct, preproc params for data.
    dataPPPrm
    
    % Timestamp, last data modification (for any reason)
    dataTS
  end
  
  %% Training Data Selection
  properties
    % Furthest-first distance threshold for training data.
    trnDataFFDThresh
    
    % Currently selected training data
    trnDataTblP
    
    trnDataSelTS
  end
  
  %%
  properties
    
    % Training state -- set during .train()
    trnRes % most recent training results
    trnResTS % timestamp for trnRes
    trnResPallMD % movie/frame metadata for trnRes.pAll
    
    % Tracking state -- set during .track()
    trkP % [NTst D T+1] reduced/pruned tracked shapes
    trkPFull % [NTst RT D T+1] Tracked shapes full data
    trkPTS % [NTst] timestamp for trkP*
    trkPMD % [NTst <ncols>] table. Movie/frame md for trkP*
    
    % View/presentation
    xyPrdCurrMovie; % [npts d nfrm] predicted labels for current Labeler movie
    hXYPrdRed; % [npts] plot handles for 'reduced' tracking results, current frame
  end
  properties (Dependent)
    nPts
  end
  
  %% Dep prop getters
  methods
    function v = get.nPts(obj)
      v = obj.lObj.nLabelPoints;
    end
  end
  
  %% Ctor/Dtor
  methods
    
    function obj = CPRLabelTracker(lObj)
      obj@LabelTracker(lObj);
    end
    
    function delete(obj)
      deleteValidHandles(obj.hXYPrdRed);
      obj.hXYPrdRed = [];
    end
    
  end
  
  %% Data
  methods
    
    function tblP = getTblPLbled(obj)
      % From .lObj, read tblP for all movies/labeledframes. Currently,
      % exclude partially-labeled frames.
      
      lObj = obj.lObj;
      [~,tblP] = CPRData.readMovsLbls(lObj.movieFilesAllFull,lObj.labeledpos,...
        lObj.labeledpostag,'lbl','noImg',true);
      
      p = tblP.p;
      tfnan = any(isnan(p),2);
      nnan = nnz(tfnan);
      if nnan>0
        warning('CPRLabelTracker:nanData','Not including %d partially-labeled rows.',nnan);
      end
      tblP = tblP(~tfnan,:);
    end
    
    function tblP = getTblP(obj,iMovs,frms)
      % From .lObj, read tblP for given movies/frames.
      
      lObj = obj.lObj;
      [~,tblP] = CPRData.readMovsLblsRaw(lObj.movieFilesAllFull,lObj.labeledpos,...
        lObj.labeledpostag,iMovs,frms,'noImg',true);
    end
    
    function [tblPnew,tblPupdate] = tblPDiff(obj,tblP)
      % Compare tblP to current data
      %
      % tblPNew: new frames 
      % tblPupdate: existing frames with new positions
      
      td = obj.data;
      tblCurrP = td.MD;
      tblCurrP.p = td.pGT;
      
      tblCurrMF = tblCurrP(:,{'mov' 'frm'});
      tblMF = tblP(:,{'mov' 'frm'});
      tfPotentiallyUpdatedRows = ismember(tblMF,tblCurrMF);
      tfNewRows = ~tfPotentiallyUpdatedRows;
      
      tblPnew = tblP(tfNewRows,:);
      tblPupdate = tblP(tfPotentiallyUpdatedRows,:);
      tblPupdate = setdiff(tblPupdate,tblCurrP);      
    end
      
    function initData(obj)
      % Initialize .data, .dataPPPrm, .dataTS
      
      I = cell(0,1);
      tblP = struct2table(struct('mov',cell(0,1),'movS',[],'frm',[],'p',[],'tfocc',[]));
      
      obj.data = CPRData(I,tblP);
      obj.dataPPPrm = [];
      obj.dataTS = now;
    end
    
    function updateData(obj,tblPNew,tblPupdate,varargin)
      % Incremental data update
      %
      % * Rows appended and pGT/tfocc updated; but other information
      % untouched
      % * PreProc parameters must be same as existing
      % * histeq (if specified in preproc params) must use existing H0,
      % unless it is the very first update (updating an empty dataset)
      % 
      %
      % tblPNew: new rows
      % tblPupdate: updated rows (rows with updated pGT/tfocc)
      %
      % sets .data, .dataPPPrm, .dataTS 
          
      [hWB] = myparse(varargin,...
        'hWaitBar',[]);

      % read/check params
      prm = obj.readParamFile();
      prmpp = prm.PreProc;
      if isempty(obj.dataPPPrm) % first update
        obj.dataPPPrm = prmpp;
      end
      if ~isequal(prmpp,obj.dataPPPrm)
        error('CPRLabelTracker:diffPrm',...
          'Cannot do incremental update; parameters have changed.');
      end

      dataCurr = obj.data;
      tblMFcurr = dataCurr.MD(:,{'mov' 'frm'});

      %%% EXISTING ROWS -- just update pGT and tfocc
      nUpdate = size(tblPupdate,1);
      if nUpdate>0 % AL 20160413 Shouldn't need to special-case, MATLAB table indexing API may not be polished
        fprintf(1,'Updating labels for %d rows...\n',nUpdate);
        tblMFupdate = tblPupdate(:,{'mov' 'frm'});
        [tf,loc] = ismember(tblMFupdate,tblMFcurr);
        assert(all(tf));
        dataCurr.MD{loc,'tfocc'} = tblPupdate.tfocc; % AL 20160413 throws if nUpdate==0
        dataCurr.pGT(loc,:) = tblPupdate.p;
      end      
      
      %%% NEW ROWS -- read images + PP
      tblMFnew = tblPNew(:,{'mov' 'frm'});
      assert(~any(ismember(tblMFnew,tblMFcurr)));      
      nNew = size(tblPNew,1);
      if nNew>0
        fprintf(1,'Adding %d new rows to data...\n',nNew);
        I = CPRData.getFrames(tblPNew);
        dataNew = CPRData(I,tblPNew);
        if prmpp.histeq
          H0 = dataCurr.H0;
          if isempty(H0)
            assert(dataCurr.N==0,'H0 can be empty only for empty/new data.');
          else
            fprintf(1,'HistEq: Using existing H0.\n');
          end
          gHE = categorical(dataNew.MD.mov);
          dataNew.histEq('g',gHE,'H0',H0,'hWaitBar',hWB);
          if isempty(H0)
            assert(~isempty(dataNew.H0));
            dataCurr.H0 = dataNew.H0; % H0s need to match for .append()
          end
        end
        if ~isempty(prmpp.channelsFcn)
          feval(prmpp.channelsFcn,dataNew,'hWaitBar',hWB);
          if isempty(dataCurr.IppInfo)
            assert(dataCurr.N==0,'Ippinfo can be empty only for empty/new data.');
            dataCurr.IppInfo = dataNew.IppInfo;
          end
        end
        
        dataCurr.append(dataNew);        
      end      
      
      obj.data = dataCurr;
      obj.dataTS = now;
    end    
    
  end
  
  %% Training Data Selection
  methods
    
    function trnDataInit(obj)
      obj.trnDataFFDThresh = nan;
      obj.trnDataTblP = [];
      obj.trnDataSelTS = -inf;
    end
      
    function trnDataSelect(obj)
      % Furthest-first selection of training data.
      % 
      % tblP: data to consider
      %
      % Based on user interaction, .trnDataFFDThresh and obj.trnDataTblP 
      % are set.      
      
      tblP = obj.getTblPLbled(); % start with all labeled data
      [grps,ffd,ffdiTrl] = CPRData.ffTrnSet(tblP,[]);

      movS = categorical(tblP.movS);
      movsUn = categories(movS);
      nMovsUn = numel(movsUn);
      movsUnCnt = countcats(movS);
      n = size(tblP,1);
      
      hFig = CPRData.ffTrnSetSelect(tblP,grps,ffd,ffdiTrl,...
        'cbkFcn',@(xSel,ySel)nst(xSel,ySel));
      
      function nst(~,ySel)
        % xSel/ySel: (x,y) on ffd plot nearest to user click (see
        % CPRData.ffTrnSetSelect)
        
        ffdThresh = ySel;
        assert(isscalar(ffd) && isscalar(ffdiTrl));
        tfSel = ffd{1}>=ffdThresh;
        iSel = ffdiTrl{1}(tfSel);
        nSel = numel(iSel);
        
        tblPSel = tblP(iSel,:);
        movsSel = categorical(tblPSel.movS);
        movsUnSelCnt = arrayfun(@(x)nnz(movsSel==x),movsUn);
        for iMov = 1:nMovsUn
          fprintf(1,'%s: nSel/nTot=%d/%d (%d%%)\n',char(movsUn(iMov)),...
            movsUnSelCnt(iMov),movsUnCnt(iMov),round(movsUnSelCnt(iMov)/movsUnCnt(iMov)*100));
        end
        fprintf(1,'Grand total of %d/%d (%d%%) shapes selected for training.\n',...
          nSel,n,round(nSel/n*100));
        
        res = input('Accept this selection (y/n/c)?','s');
        if isempty(res)
          res = 'c';
        end
        switch lower(res)
          case 'y'
            obj.trnDataFFDThresh = ffdThresh;
            obj.trnDataTblP = tblPSel;
            obj.trnDataSelTS = now;
          case 'n'
            % none
          case 'c'
            delete(hFig);
        end
      end
    end

  end  
  
  %%
  methods
   
    function initHook(obj)
      obj.initData();
      obj.trnDataInit();
      
      obj.trnRes = [];
      obj.trnResPallMD = [];
      obj.trnResTS = [];
      
      obj.trkP = [];
      obj.trkPFull = [];
      obj.trkPTS = zeros(0,1);
      obj.trkPMD = struct2table(struct('mov',cell(0,1),'movS',[],'frm',[]));
            
      obj.xyPrdCurrMovie = [];
      deleteValidHandles(obj.hXYPrdRed);
      obj.hXYPrdRed = [];
      
      npts = obj.nPts;
      ptsClrs = obj.lObj.labelPointsPlotInfo.Colors;
      plotPrefs = obj.lObj.trackPrefs.PredictPointsPlot;
      ax = obj.ax;
      cla(ax);
      hold(ax,'on');
      hTmp = gobjects(npts,1);
      for iPt = 1:npts
        clr = ptsClrs(iPt,:);
        hTmp(iPt) = plot(ax,nan,nan,plotPrefs.Marker,...
          'MarkerSize',plotPrefs.MarkerSize,...
          'LineWidth',plotPrefs.LineWidth,...
          'Color',clr);
      end      
      obj.hXYPrdRed = hTmp;      
    end
        
    function prm = readParamFile(obj)
      prmFile = obj.paramFile;
      if isempty(prmFile)
        error('CPRLabelTracker:noParams',...
          'Tracking parameter file needs to be set.');
      end
      prm = ReadYaml(prmFile);
    end
    
    function train(obj)
      prm = obj.readParamFile();
      
      tblPTrn = obj.trnDataTblP;
      if isempty(tblPTrn)
        error('CPRLabelTracker:noTrnData','No training data selected.');
      end
      [tblPnew,tblPupdate] = obj.tblPDiff(tblPTrn);
      
      hWB = waitbar(0);
      hTxt = findall(hWB,'type','text');
      hTxt.Interpreter = 'none';
      obj.updateData(tblPnew,tblPupdate,'hWaitBar',hWB);
      
      d = obj.data;
      tblMF = d.MD(:,{'mov' 'frm'});
      tblTrnMF = tblPTrn(:,{'mov','frm'});
      tf = ismember(tblMF,tblTrnMF);
      d.iTrn = find(tf);
      
      d.summarize('movS',d.iTrn);
            
      [Is,nChan] = d.getCombinedIs(d.iTrn);
      prm.Ftr.nChn = nChan;
      
      delete(hWB); % AL: get this guy in training?
      
      tr = train(d.pGTTrn,d.bboxesTrn,Is,...
          'modelPrms',prm.Model,...
          'regPrm',prm.Reg,...
          'ftrPrm',prm.Ftr,...
          'initPrm',prm.TrainInit,...
          'prunePrm',prm.Prune,...
          'docomperr',false,...
          'singleoutarg',true);
      obj.trnRes = tr;
      obj.trnResTS = now;
      obj.trnResPallMD = d.MD;
    end
    
    function inspectTrainingData(obj)
      d = obj.data;
      if d.NTrn==0
        error('CPRLabelTracker:noTD','No training data is available.');
      end
      d.vizWithFurthestFirst();      
    end
    
    function track(obj,iMovs,frms)
      if isempty(obj.trnRes)
        error('CPRLabelTracker:noRes','No tracker has been trained.');
      end
      
      prm = obj.readParamFile();

      hWB = waitbar(0);
      hTxt = findall(hWB,'type','text');
      hTxt.Interpreter = 'none';

      tblP = obj.getTblP(iMovs,frms);
      [tblPnew,tblPupdate] = obj.tblPDiff(tblP);
      obj.updateData(tblPnew,tblPupdate,'hWaitBar',hWB);
      d = obj.data;
      
      tblMFTrk = tblP(:,{'mov' 'frm'});
      tblMFAll = d.MD(:,{'mov' 'frm'});
      [tf,loc] = ismember(tblMFTrk,tblMFAll);
      assert(all(tf));
      d.iTst = loc;
      
      d.summarize('movS',d.iTst);

      delete(hWB);
 
      [Is,nChan] = d.getCombinedIs(d.iTst);
      prm.Ftr.nChn = nChan;
            
      %% Test on test set
      tr = obj.trnRes;
      prmInit = prm.TestInit;
      NTst = d.NTst;
      RT = prmInit.Nrep;
      Tp1 = tr.regModel.T+1;
      mdl = tr.regModel.model;
      
      pGTTrnNMu = nanmean(tr.regModel.pGtN,1);
      pIni = shapeGt('initTest',[],d.bboxesTst,mdl,[],...
        repmat(pGTTrnNMu,NTst,1),RT,prmInit.augrotate);
      VERBOSE = 0;
      [~,p_t] = rcprTest1(Is,tr.regModel,pIni,tr.regPrm,tr.ftrPrm,...
        d.bboxesTst,VERBOSE,tr.prunePrm);
      pTstT = reshape(p_t,[NTst RT mdl.D Tp1]);      
      
      %% Select best preds for each time
      pTstTRed = nan(NTst,mdl.D,Tp1);
      prunePrm = tr.prunePrm;
      prunePrm.prune = 1;
      for t = 1:Tp1
        fprintf('Pruning t=%d\n',t);
        pTmp = permute(pTstT(:,:,:,t),[1 3 2]); % [NxDxR]
        pTstTRed(:,:,t) = rcprTestSelectOutput(pTmp,tr.regModel.model,prunePrm);
      end
      
      % Augment .trkP* state with new tracking results
      % - new rows are just added
      % - existing rows are overwritten
      trkPMDnew = d.MDTst(:,{'mov' 'movS' 'frm'});
      trkPMDcur = obj.trkPMD;
      [tf,loc] = ismember(trkPMDnew,trkPMDcur);
      % existing rows
      idxCur = loc(tf);
      obj.trkP(idxCur,:,:) = pTstTRed(tf,:,:);
      obj.trkPFull(idxCur,:,:,:) = pTstT(tf,:,:,:);
      nowts = now;
      obj.trkPTS(idxCur) = nowts;
      % new rows
      obj.trkP = [obj.trkP; pTstTRed(~tf,:,:)];
      obj.trkPFull = [obj.trkPFull; pTstT(~tf,:,:,:)];
      nNew = nnz(~tf);
      obj.trkPTS = [obj.trkPTS; repmat(nowts,nNew,1)];
      obj.trkPMD = [obj.trkPMD; trkPMDnew(~tf,:)];
      
      obj.loadXYPrdCurrMovie();
      obj.newLabelerFrame();      

      %       if ~skipLoss        
%         %%
%         hFig = Shape.vizLossOverTime(td.pGTTst,pTstTRed,'md',td.MDTst);
%         
%         %%
%         hFig(end+1) = figure('WindowStyle','docked');
%         iTst = td.iTst;
%         tfTstLbled = ismember(iTst,find(td.isFullyLabeled));
%         Shape.vizDiff(td.ITst(tfTstLbled),td.pGTTst(tfTstLbled,:),...
%           pTstTRed(tfTstLbled,:,end),tr.regModel.model,...
%           'fig',gcf,'nr',4,'nc',4,'md',td.MDTst(tfTstLbled,:));
%       end            
    end
    
    function loadXYPrdCurrMovie(obj)
      % sets .xyPrdCurrMovie for current Labeler movie from .trkP, .trkPMD 
      
      trkTS = obj.trkPTS;
      if isempty(trkTS)
        obj.xyPrdCurrMovie = [];
        return;
      end
      
      if any(trkTS<obj.trnResTS) || any(trkTS<obj.dataTS)
        warning('CPRLabelTracker:trackOOD',...
          'Some/all tracking results may be out of date.');
      end
        
      lObj = obj.lObj;
      movName = lObj.movieFilesAllFull{lObj.currMovie};
      nfrms = lObj.nframes;
      
      mdl = obj.trnRes.regModel.model;
      pTrk = obj.trkP(:,:,end);
      trkMD = obj.trkPMD;
      assert(isequal(size(pTrk),[size(trkMD,1) mdl.D]));
      
      xy = nan(mdl.nfids,mdl.d,nfrms);
      tfCurrMov = strcmp(trkMD.mov,movName); % these rows of trnData/MD are for the current Labeler movie
      nCurrMov = nnz(tfCurrMov);
      xyTrkCurrMov = reshape(pTrk(tfCurrMov,:)',mdl.nfids,mdl.d,nCurrMov); % [npt x d x nCurrMov]
      
      frmCurrMov = trkMD.frm(tfCurrMov);
      xy(:,:,frmCurrMov) = xyTrkCurrMov;
      obj.xyPrdCurrMovie = xy;
    end
      
    function newLabelerFrame(obj)
      % Update .hXYPrdRed based on current Labeler frame and
      % .xyPrdCurrMovie
      
      frm = obj.lObj.currFrame;
      npts = obj.nPts;
      hXY = obj.hXYPrdRed;
      if isempty(obj.xyPrdCurrMovie)
        xy = nan(npts,2);
      else
        xy = obj.xyPrdCurrMovie(:,:,frm); % [npt x d]
      end
      for iPt = 1:npts
        set(hXY(iPt),'XData',xy(iPt,1),'YData',xy(iPt,2));
      end
    end
    
    function newLabelerMovie(obj)
      if isempty(obj.trnRes)
        return;
      end
      
      obj.loadXYPrdCurrMovie();
      obj.newLabelerFrame();      
    end
    
    function s = getSaveToken(obj)
      s = struct();
      s.labelTrackerClass = class(obj);
      
      d = obj.data;
      tblP = d.MD;
      tblP.p = d.pGT;
      s.tblP = tblP;
      for p = obj.TOKEN_SAVEPROPS, p=p{1}; %#ok<FXSET>
        s.(p) = obj.(p);
      end
    end
    
    function loadSaveToken(obj,s)
      assert(isequal(s.labelTrackerClass,class(obj)));      
      
      for p = obj.TOKEN_LOADPROPS, p=p{1}; %#ok<FXSET>
        obj.(p) = s.(p);
      end
      
      obj.loadXYPrdCurrMovie();
      obj.newLabelerFrame();
    end
    
  end
  
  methods (Static)
        
    function tdPPJan(td,varargin)
      td.computeIpp([],[],[],'iTrl',1:td.N,'jan',true,varargin{:});
    end
    
  end
  
end