classdef CPRLabelTracker < LabelTracker
  
  properties (Constant,Hidden)
    TRAINEDTRACKER_SAVEPROPS = { ...
      'sPrm' ...
      'trnDataFFDThresh' 'trnDataTblP' 'trnDataTblPTS' ...
      'trnResIPt' 'trnResRC'};
    TRACKRES_SAVEPROPS = {'trkP' 'trkPFull' 'trkPTS' 'trkPMD' 'trkPiPt'};
  end
  
  %% Params
  properties
    sPrm % full parameter struct
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
    
    % Timestamp, last data modification (for any reason)
    dataTS
  end
  
  %% Training Data Selection
  properties
    % Furthest-first distance threshold for training data.
    trnDataFFDThresh
    
    % Currently selected training data (includes updates/additions)
    trnDataTblP
    trnDataTblPTS % [size(trnDataTblP,1)x1] timestamps for when rows of trnDataTblP were added to CPRLabelTracker
  end
  
  %% Train/Track
  properties
    
    % Training state -- set during .train()
%     trnRes % most recent training results
%     trnResTS % timestamp for trnRes
%     trnResPallMD % movie/frame metadata for trnRes.pAll
    trnResIPt %
    trnResRC % RegressorCascade
    
    % Tracking state -- set during .track()
    % Note: trkD here can be less than the full/model D if some pts are
    % omitted from tracking
    trkP % [NTst trkD T+1] reduced/pruned tracked shapes
    trkPFull % [NTst RT trkD T+1] Tracked shapes full data
    trkPTS % [NTst] timestamp for trkP*
    trkPMD % [NTst <ncols>] table. Movie/frame md for trkP*
    trkPiPt % [trkD] indices into 1:model.D, tracked points
    
    % View/presentation
    xyPrdCurrMovie; % [npts d nfrm] predicted labels for current Labeler movie
    xyPrdCurrMovieIsInterp; % [nfrm] logical vec indicating whether xyPrdCurrMovie(:,:,i) is interpolated
    hXYPrdRed; % [npts] plot handles for 'reduced' tracking results, current frame
    xyVizPlotArgs; % cell array of args for regular tracking viz
    xyVizPlotArgsInterp; % cell array of args for interpolated tracking viz
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
    
    % AL 20160531 Data and timestamps
    %
    % - All labels are timestamped in the labeler (each pt has its own ts).-
    % This is done at manual-label-time (eg clicktime). The .pTS field in
    % MD tables contains this ts.
    % - .trnDataTblPTS contains timestamps labeling rows of .trnDataTblP.
    % These timestamps label when labels are accepted into the training
    % set.
    %
    % It is necessary to track this timestamp b/c the selection of a 
    % training set represents a filtering of labeled data, ie some labeled 
    % data will not be included. Once a training data set is selected,
    % moving forward we may want to find new labels made since the time of
    % that training set selection etc.
    %
    % - The RegressorCascade in .trnResRC has its own timestamps for
    % training time and so on.
    
    function tblP = getTblPLbledRecent(obj)
      % tblP: labeled data from Labeler that is more recent than anything in .trnDataTblPTS
      
      tblP = obj.getTblPLbled();
      maxTS = max(tblP.pTS,[],2);
      maxTDTS = max([obj.trnDataTblPTS(:);-inf]);
      tf = maxTS > maxTDTS;
      tblP = tblP(tf,:);
    end 
    
    function tblP = getTblP(obj,iMovs,frms)
      % From .lObj, read tblP for given movies/frames.
      
      lObj = obj.lObj;
      [~,tblP] = Labeler.lblCompileContentsRaw(lObj.movieFilesAllFull,lObj.labeledpos,...
        lObj.labeledpostag,iMovs,frms,'noImg',true,'lposTS',lObj.labeledposTS);
    end
    
    function [tblPnew,tblPupdate] = tblPDiffData(obj,tblP)
      td = obj.data;
      tbl0 = td.MD;
      tbl0.p = td.pGT;
      [tblPnew,tblPupdate] = CPRLabelTracker.tblPDiff(tbl0,tblP);
    end
    
    function [tblPnew,tblPupdate,idxTrnDataTblP] = tblPDiffTrnData(obj,tblP)
      [tblPnew,tblPupdate,idxTrnDataTblP] = CPRLabelTracker.tblPDiff(obj.trnDataTblP,tblP);
    end
    
  end
  
  methods (Static)
    function [tblPnew,tblPupdate,idx0update] = tblPDiff(tblP0,tblP)
      % Compare tblP to tblP0 
      %
      % tblP0, tblP: MD/p tables
      %
      % tblPNew: new frames (rows of tblP whose movie-frame ID are not in tblP0)
      % tblPupdate: existing frames with new positions/tags (rows of tblP 
      %   whos movie+frame ID are in tblP0, but whose eg p field is different).
      % idx0update: indices into rows of tblP0 corresponding to tblPupdate;
      %   ie tblP0(idx0update,:) ~ tblPupdate
            
      tblMF0 = tblP0(:,{'mov' 'frm'});
      tblMF = tblP(:,{'mov' 'frm'});
      tfPotentiallyUpdatedRows = ismember(tblMF,tblMF0);
      tfNewRows = ~tfPotentiallyUpdatedRows;
      
      tblPnew = tblP(tfNewRows,:);
      tblPupdate = tblP(tfPotentiallyUpdatedRows,:);
      tblPupdate = setdiff(tblPupdate,tblP0);
      
      [tf,loc] = ismember(tblPupdate(:,{'mov' 'frm'}),tblMF0);
      assert(all(tf));
      idx0update = loc(tf);
    end
  end
  
  methods
    
    function initData(obj)
      % Initialize .data*
      
      I = cell(0,1);
      tblP = struct2table(struct('mov',cell(0,1),'movS',[],'frm',[],'p',[],'tfocc',[]));
      
      obj.data = CPRData(I,tblP);
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
      % sets .data, .dataTS
      
      [hWB] = myparse(varargin,...
        'hWaitBar',[]);
      
      prmpp = obj.sPrm.PreProc;
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
      
      if nUpdate>0 || nNew>0      
        obj.data = dataCurr;
        obj.dataTS = now;
      else
        warningNoTrace('CPRLabelTracker:data','Nothing to update in data.');
      end
    end
    
  end
  
  %% Training Data Selection
  methods
    
    function trnDataInit(obj)
      obj.trnDataFFDThresh = nan;
      obj.trnDataTblP = [];
      obj.trnDataTblPTS = -inf(0,1);
    end
    
    function trnDataSelect(obj)
      % Furthest-first selection of training data.
      %
      % Based on user interaction, .trnDataFFDThresh, .trnDataTblP* are set.
      % For .trnDataTblP*, this is a fresh reset, not an update.
      
      if ~isempty(obj.trnResRC) && obj.trnResRC.hasTrained
        resp = questdlg('A tracker has already been trained. Re-selecting training data will clear all previous trained/tracked results. Proceed?',...
          'Tracker Trained','Yes, clear previous tracker','Cancel','Cancel');
        if isempty(resp)
          resp = 'Cancel';
        end
        switch resp
          case 'Yes, clear previous tracker'
            obj.trnResInit();
            obj.trackResInit();
            obj.vizInit();
          case 'Cancel'
            return;
        end
      end
        
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
            obj.trnDataTblPTS = now*ones(size(tblPSel,1),1);
          case 'n'
            % none
          case 'c'
            delete(hFig);
        end
      end
    end
    
  end
  
  %% TrainRes
  methods
    function trnResInit(obj)
      if isempty(obj.sPrm)
        obj.trnResRC = [];
      else
        obj.trnResRC = RegressorCascade(obj.sPrm);
      end
      obj.trnResIPt = [];
    end
  end
  
  %% LabelTracker overloads
  methods
    
    function initHook(obj)
      obj.initData();
      obj.trnDataInit();
      obj.trnResInit();
      obj.trackResInit();
      obj.vizInit();
    end
    
    function setParamHook(obj)
      sNew = obj.readParamFileYaml();
      sOld = obj.sPrm;      
      obj.sPrm = sNew; % set this now so eg trnResInit() can use
      
      if isempty(sOld)
        obj.trnResInit();
        obj.trackResInit();
        obj.vizInit();
      else
        % Figure out what changed
        flds = fieldnames(sOld);
        tfunchanged = struct();
        for f=flds(:)',f=f{1}; %#ok<FXSET>
          tfunchanged.(f) = isequaln(sOld.(f),sNew.(f));
        end
        
        % data
        modelPPUC = tfunchanged.Model && tfunchanged.PreProc;
        if ~modelPPUC
          fprintf(2,'Parameter change: CPRLabelTracker data cleared.\n');
          obj.initData();
        end
        
        % trainingdata
        if ~modelPPUC
          fprintf(2,'Parameter change: CPRLabelTracker training data selection cleared.\n');
          obj.trnDataInit();
        end
      
        % trnRes
        modelPPRegFtrTrnInitUC = modelPPUC && tfunchanged.Reg ...
            && tfunchanged.Ftr && tfunchanged.TrainInit;
        if ~modelPPRegFtrTrnInitUC
          fprintf(2,'Parameter change: CPRLabelTracker regressor casacade cleared.\n');
          obj.trnResInit();
        end
      
        % trkP
        if ~(modelPPRegFtrTrnInitUC && tfunchanged.TestInit && tfunchanged.Prune)
          fprintf(2,'Parameter change: CPRLabelTracker tracking results cleared.\n');
          obj.trackResInit();
          obj.vizInit();
        end
      end
    end
    
    function retrain(obj,varargin)
      % Full train using all of obj.trnDataTblP as training data
      % 
      % Sets .trnRes*
      
      useRC = myparse(varargin,...
        'useRC',true... % always true now (use RegressorCascade)
        );
      
      prm = obj.sPrm;
      
      tblPTrn = obj.trnDataTblP;
      if isempty(tblPTrn)
        error('CPRLabelTracker:noTrnData','No training data selected.');
      end
      tblPTrn(:,'pTS') = [];
      [tblPnew,tblPupdate] = obj.tblPDiffData(tblPTrn);
      
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
      
      iPt = prm.TrainInit.iPt;
      nfids = prm.Model.nfids;
      assert(prm.Model.d==2);
      nfidsInTD = size(d.pGT,2)/prm.Model.d;
      if isempty(iPt)
        assert(nfidsInTD==nfids);
        iPt = 1:nfidsInTD;
      end
      iPGT = [iPt iPt+nfidsInTD];
      fprintf(1,'iPGT: %s\n',mat2str(iPGT));
      
      pTrn = d.pGTTrn(:,iPGT);
      if useRC
        obj.trnResRC.trainWithRandInit(Is,d.bboxesTrn,pTrn);
      else
        assert(false,'Unsupported');
%         tr = train(pTrn,d.bboxesTrn,Is,...
%           'modelPrms',prm.Model,...
%           'regPrm',prm.Reg,...
%           'ftrPrm',prm.Ftr,...
%           'initPrm',prm.TrainInit,...
%           'prunePrm',prm.Prune,...
%           'docomperr',false,...
%           'singleoutarg',true);
%         obj.trnRes = tr;
      end
%       obj.trnResTS = now;
%       obj.trnResPallMD = d.MD;
      obj.trnResIPt = iPt;
    end
    
    function train(obj,varargin)
      % Incremental trainupdate using labels newer than .trnDataTblPTS

      % figure out if we want an incremental train or full retrain
      rc = obj.trnResRC;
      if ~rc.hasTrained
        obj.retrain(varargin{:});
        return;
      end        
            
      prm = obj.sPrm;
      tblPNew = obj.getTblPLbledRecent();
      
      if isempty(tblPNew)
        msgbox('Trained tracker is up-to-date with labels.','Tracker up-to-date');
        return;
      end
      
      %%% do incremental train 
      
      % update the TrnData
      [tblPNewTD,tblPUpdateTD,idxTrnDataTblP] = obj.tblPDiffTrnData(tblPNew);
      obj.trnDataTblP(idxTrnDataTblP,:) = tblPUpdateTD;
      obj.trnDataTblP = [obj.trnDataTblP; tblPNewTD];
      nowtime = now();
      obj.trnDataTblPTS(idxTrnDataTblP) = nowtime;
      obj.trnDataTblPTS = [obj.trnDataTblPTS; nowtime*ones(size(tblPNewTD,1),1)];
      
      % print out diagnostics on when training occurred etc
      iTL = rc.trnLogMostRecentTrain();
      tsFullTrn = rc.trnLog(iTL).ts;
      fprintf('Most recent full train at %s\n',datestr(tsFullTrn,'mmm-dd-yyyy HH:MM:SS'));
      obj.trainPrintDiagnostics(iTL);
     
      % update the data
      tblPNew(:,'pTS') = [];
      [tblPnew,tblPupdate] = obj.tblPDiffData(tblPNew);
      obj.updateData(tblPnew,tblPupdate);
      
      % set iTrn and summarize
      d = obj.data;
      tblMF = d.MD(:,{'mov' 'frm'});
      tblNewMF = tblPNew(:,{'mov','frm'});
      tf = ismember(tblMF,tblNewMF);
      assert(nnz(tf)==size(tblNewMF,1));
      d.iTrn = find(tf);
      d.summarize('movS',d.iTrn);
      
      % Call rc.train with 'update', true 
      [Is,nChan] = d.getCombinedIs(d.iTrn);
      prm.Ftr.nChn = nChan;
            
      iPt = prm.TrainInit.iPt;
      nfids = prm.Model.nfids;
      assert(prm.Model.d==2);
      nfidsInTD = size(d.pGT,2)/prm.Model.d;
      if isempty(iPt)
        assert(nfidsInTD==nfids);
        iPt = 1:nfidsInTD;
      end
      iPGT = [iPt iPt+nfidsInTD];
      fprintf(1,'iPGT: %s\n',mat2str(iPGT));
      
      pTrn = d.pGTTrn(:,iPGT);

      rc = obj.trnResRC;
      rc.trainWithRandInit(Is,d.bboxesTrn,pTrn,'update',true,'initpGTNTrn',true);

      %obj.trnResTS = now;
      %obj.trnResPallMD = d.MD;
      assert(isequal(obj.trnResIPt,iPt));
    end
    
    function trainPrintDiagnostics(obj,iTL)
      % iTL: Index into .trnLog at which to start
      
      rc = obj.trnResRC;
      tsFullTrn = rc.trnLog(iTL).ts;

      nTL = numel(rc.trnLog);
      tsTrnDataUn = unique(obj.trnDataTblPTS);
      tsTrnDataUn = tsTrnDataUn(tsTrnDataUn>=tsFullTrn); % trn data updates after most recent fulltrain
      tsTrnDataUn = sort(tsTrnDataUn);
      ntsTrnData = numel(tsTrnDataUn);
      itsTD = 0;
      while iTL<nTL || itsTD<ntsTrnData
        if iTL==nTL
          action = 'trnData';
        elseif itsTD==ntsTrnData
          action = 'trnLog';
        elseif rc.trnLog(iTL+1).ts<tsTrnDataUn(itsTD+1)
          action = 'trnLog';
        else
          action = 'trnData';
        end
        
        switch action
          case 'trnLog'
            fprintf('%s at %s\n',rc.trnLog(iTL+1).action,...
              datestr(rc.trnLog(iTL+1).ts,'mmm-dd-yyyy HH:MM:SS'));
            iTL = iTL+1;
          case 'trnData'
            ntmp = nnz(obj.trnDataTblPTS==tsTrnDataUn(itsTD+1));
            fprintf('%d labels added to training data at %s\n',ntmp,...
              datestr(tsTrnDataUn(itsTD+1),'mmm-dd-yyyy HH:MM:SS'));
            itsTD = itsTD+1;
        end
      end
    end
    
    % MOVE THIS METHOD BELOW
    function loadTrackResMerge(obj,fname)
      % Load tracking results from fname, merging into existing results
      
      tr = load(fname);
      if ~isempty(obj.paramFile) && ~strcmp(tr.paramFile,obj.paramFile)
        warningNoTrace('CPRLabelTracker:paramFile',...
          'Tracking results generated using parameter file ''%s'', which differs from current file ''%s''.',...
          tr.paramFile,obj.paramFile);
      end
      
      if ~isempty(obj.trkP) % training results exist
        
        if ~isequal(obj.trkPiPt,tr.trkPiPt)
          error('CPRLabelTracker:trkPiPt','''trkPiPt'' differs in tracked results to be loaded.');
        end
        
        tblMF = obj.trkPMD(:,{'movS' 'frm'});
        tblLoad = tr.trkPMD(:,{'movS' 'frm'});
        [tfOverlp,locMF] = ismember(tblLoad,tblMF);
        
        tsOverlp0 = obj.trkPTS(locMF(tfOverlp));
        tsOverlpNew = tr.trkPTS(tfOverlp);
        nOverlapOlder = nnz(tsOverlpNew<tsOverlp0);
        if nOverlapOlder>0
          warningNoTrace('CPRLabelTracker:trkPTS',...
            'Loading tracking results that are older than current results for %d frames.',nOverlapOlder);
        end
        
        % load existing/overlap results
        iOverlp = locMF(tfOverlp);
        obj.trkP(iOverlp,:,:) = tr.trkP(tfOverlp,:,:);
        obj.trkPFull(iOverlp,:,:,:) = tr.trkPFull(tfOverlp,:,:,:);
        obj.trkPMD(iOverlp,:) = tr.trkPMD(tfOverlp,:);
        obj.trkPTS(iOverlp,:) = tr.trkPTS(tfOverlp,:);
        
        % load new results
        obj.trkP = cat(1,obj.trkP,tr.trkP(~tfOverlp,:,:));
        obj.trkPFull = cat(1,obj.trkPFull,tr.trkPFull(~tfOverlp,:,:,:));
        obj.trkPMD = cat(1,obj.trkPMD,tr.trkPMD(~tfOverlp,:));
        obj.trkPTS = cat(1,obj.trkPTS,tr.trkPTS(~tfOverlp,:));
      else
        % code in the other branch would basically work, but we also want
        % to set trkPiPt
        props = obj.TRACKRES_SAVEPROPS;
        for p=props(:)',p=p{1}; %#ok<FXSET>
          obj.(p) = tr.(p);
        end
      end
      
      nfLoad = size(tr.trkP,1);
      fprintf(1,'Loaded tracking results for %d frames.\n',nfLoad);
    end
    
    function track(obj,iMovs,frms,varargin)
      [useRC,tblP] = myparse(varargin,...
        'useRC',true,... % if true, use RegressorCascade (.trnResRC)
        'tblP',[]... % table with props {'mov' 'frm' 'p'} containing movs/frms to track
        );
      
      prm = obj.sPrm;
      
      hWB = waitbar(0);
      hTxt = findall(hWB,'type','text');
      hTxt.Interpreter = 'none';
      
      if isempty(tblP)
        tblP = obj.getTblP(iMovs,frms);
      end
      tblP(:,'pTS') = [];
      [tblPnew,tblPupdate] = obj.tblPDiffData(tblP);
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
      NTst = d.NTst;
      RT = prm.TestInit.Nrep;
      bboxes = d.bboxesTst;
      
      if useRC
        rc = obj.trnResRC;
        p_t = rc.propagateRandInit(Is,bboxes,prm.TestInit);
        trkMdl = rc.prmModel;
        trkD = trkMdl.D;
        Tp1 = rc.nMajor+1;
      else
        assert(false,'Unsupported');
%         tr = obj.trnRes;
%         if isempty(tr)
%           error('CPRLabelTracker:noRes','No tracker has been trained.');
%         end
%         Tp1 = tr.regModel.T+1;
%         trkMdl = tr.regModel.model;
%         trkD = trkMdl.D;
%         
%         pGTTrnNMu = nanmean(tr.regModel.pGtN,1);
%         pIni = shapeGt('initTest',[],bboxes,trkMdl,[],...
%           repmat(pGTTrnNMu,NTst,1),RT,prmInit.augrotate);
%         VERBOSE = 0;
%         [~,p_t] = rcprTest1(Is,tr.regModel,pIni,tr.regPrm,tr.ftrPrm,...
%           bboxes,VERBOSE,tr.prunePrm);
      end
      pTstT = reshape(p_t,[NTst RT trkD Tp1]);
      
      
      %% Select best preds for each time
      pTstTRed = nan(NTst,trkD,Tp1);
      prm.Prune.prune = 1;
      for t = 1:Tp1
        fprintf('Pruning t=%d\n',t);
        pTmp = permute(pTstT(:,:,:,t),[1 3 2]); % [NxDxR]
        pTstTRed(:,:,t) = rcprTestSelectOutput(pTmp,trkMdl,prm.Prune);
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
      obj.trkPiPt = obj.trnResIPt;
      
      if ~isempty(obj.lObj)
        obj.vizLoadXYPrdCurrMovie();
        obj.newLabelerFrame();
      else
        % headless mode
      end
      
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
        
    function newLabelerFrame(obj)
      % Update .hXYPrdRed based on current Labeler frame and
      % .xyPrdCurrMovie
      
      % get xy and isinterp
      frm = obj.lObj.currFrame;
      npts = obj.nPts;
      if isempty(obj.xyPrdCurrMovie)
        xy = nan(npts,2);
        isinterp = false;
      else
        % AL20160502: When changing movies, order of updates to 
        % lObj.currMovie and lObj.currFrame is unspecified. currMovie can
        % be updated first, resulting in an OOB currFrame; protect against
        % this.
        frm = min(frm,size(obj.xyPrdCurrMovie,3));
        
        xy = obj.xyPrdCurrMovie(:,:,frm); % [npt x d]
        isinterp = obj.xyPrdCurrMovieIsInterp(frm);
      end
      
      if isinterp
        plotargs = obj.xyVizPlotArgsInterp;
      else
        plotargs = obj.xyVizPlotArgs;
      end      
      
      hXY = obj.hXYPrdRed;
      for iPt = 1:npts
        set(hXY(iPt),'XData',xy(iPt,1),'YData',xy(iPt,2),plotargs{:});
      end
    end
    
    function newLabelerMovie(obj)
      obj.vizLoadXYPrdCurrMovie();
      obj.newLabelerFrame();
    end
    
    function s = getSaveToken(obj)
      % See save philosophy below. ATM we return a "full" struct with
      % 1(subset)+2+3+4;
      
      s1 = obj.getTrainedTrackerSaveStruct();
      s2 = obj.getTrackResSaveStruct();
      if isfield(s1,'paramFile') && isfield(s2,'paramFile')
        assert(isequal(s1.paramFile,s2.paramFile));
        s2 = rmfield(s2,'paramFile');
      end
      s = structmerge(s1,s2);
      s.labelTrackerClass = class(obj);
    end
    
    function loadSaveToken(obj,s)
      assert(strcmp(s.labelTrackerClass,class(obj)));
      s = rmfield(s,'labelTrackerClass');
      
      % ATM just load all fields
      
      assert(isfield(s,'paramFile'));
      if ~isequal(s.paramFile,obj.paramFile)
        warningNoTrace('CPRLabelTracker:paramFile',...
          'Setting parameter file to ''%s''.',s.paramFile);
      end
      
      flds = fieldnames(s);
      for f=flds(:)',f=f{1}; %#ok<FXSET>
        obj.(f) = s.(f);
      end
      
      obj.vizLoadXYPrdCurrMovie();
      obj.newLabelerFrame();
    end
    
  end
    
  %% Save, Load, Init etc
  % The serialization philosophy is as follows.
  %
  % At a high level there are four groups of state forming a linear
  % dependency chain (basically)
  % 0. sPrm: all state is dependent on parameters.
  % 1. (CPR)Data: .data, .dataTS.
  % 2. Training Data specification: .trnData*
  % 3. Training results (trained tracker): .trnRes*
  % 4: Tracking results: .trkP*
  %
  % - Often, the .data itself in 1) will be very large. So ATM we do not
  % save the .data itself.
  % - You might want to save just 2., but much more commonly you will want
  % to save 2+3. If you want to save 3, it really makes sense to save 2 as
  % well. So we support saving 1(subset)+2+3. This is saving/loading a 
  % "trained tracker".
  % - You might want to save 4. You might want to do this independent of
  % 1(subset)+2+3. So we support saving 4 orthogonally, although of course
  % sometimes you will want to save everything.
  %
  methods
    
    % AL 20160530 all these meths need review after sPrm, various cleanups
    
    function s = getTrainedTrackerSaveStruct(obj)
      s = struct();
      props = obj.TRAINEDTRACKER_SAVEPROPS;
      s.paramFile = obj.paramFile;
      for p=props(:)',p=p{1}; %#ok<FXSET>
        s.(p) = obj.(p);
      end
    end
    
    function loadTrainedTrackerSaveStruct(obj,s)
      if ~isempty(obj.paramFile) && ~isequal(s.paramFile,obj.paramFile)
        warningNoTrace('CPRLabelTracker:paramFile',...
          'Tracker trained using parameter file ''%s'', which differs from current file ''%s''.',...
          s.paramFile,obj.paramFile);
      end      
      
      obj.paramFile = s.paramFile;
      props = obj.TRAINEDTRACKER_SAVEPROPS;
      for p=props(:)',p=p{1}; %#ok<FXSET>
        obj.(p) = s.(p);
      end
    end
    
    function saveTrainedTracker(obj,fname)
      s = obj.getTrainedTrackerSaveStruct(); %#ok<NASGU>
      save(fname,'-mat','-struct','s');
    end
    
    function loadTrainedTracker(obj,fname)
      s = load(fname,'-mat');
      obj.loadTrainedTrackerSaveStruct(s);
    end
    
    function s = getTrackResSaveStruct(obj)
      s = struct();
      s.paramFile = obj.paramFile;
      props = obj.TRACKRES_SAVEPROPS;
      for p=props(:)',p=p{1}; %#ok<FXSET>
        s.(p) = obj.(p);
      end
    end
    
    function loadTrackResSaveStruct(obj,s)
      if ~isempty(obj.paramFile) && ~isequal(s.paramFile,obj.paramFile)
        warningNoTrace('CPRLabelTracker:paramFile',...
          'Results tracked using parameter file ''%s'', which differs from current file ''%s''.',...
          s.paramFile,obj.paramFile);
      end
      obj.paramFile = s.paramFile;
      props = obj.TRACKRES_SAVEPROPS;
      for p=props(:)',p=p{1}; %#ok<FXSET>
        obj.(p) = s.(p);
      end
    end
    
    function saveTrackRes(obj,fname)
      s = obj.getTrackResSaveStruct(); %#ok<NASGU>
      save(fname,'-mat','-struct','s');
    end
    
    function trackResInit(obj)
      % init obj.TRACKRES_SAVEPROPS
      
      obj.trkP = [];
      obj.trkPFull = [];
      obj.trkPTS = zeros(0,1);
      obj.trkPMD = struct2table(struct('mov',cell(0,1),'movS',[],'frm',[]));
      obj.trkPiPt = [];
    end
    
  end
  
  %% Viz
  methods

    function vizInit(obj)
      obj.xyPrdCurrMovie = [];
      obj.xyPrdCurrMovieIsInterp = [];
      deleteValidHandles(obj.hXYPrdRed);
      obj.hXYPrdRed = [];
      
      % init .xyVizPlotArgs*
      trackPrefs = obj.lObj.trackPrefs;
      plotPrefs = trackPrefs.PredictPointsPlot;
      obj.xyVizPlotArgs = struct2paramscell(plotPrefs);
      if isfield(trackPrefs,'PredictInterpolatePointsPlot')
        obj.xyVizPlotArgsInterp = struct2paramscell(trackPrefs.PredictInterpolatePointsPlot);
      else
        obj.xyVizPlotArgsInterp = obj.xyVizPlotArgs;
      end      
      
      npts = obj.nPts;
      ptsClrs = obj.lObj.labelPointsPlotInfo.Colors;
      ax = obj.ax;
      cla(ax);
      hold(ax,'on');
      hTmp = gobjects(npts,1);
      for iPt = 1:npts
        clr = ptsClrs(iPt,:);
        hTmp(iPt) = plot(ax,nan,nan,obj.xyVizPlotArgs{:},'Color',clr);
      end
      obj.hXYPrdRed = hTmp;
    end
    
    function vizLoadXYPrdCurrMovie(obj)
      % sets .xyPrdCurrMovie* for current Labeler movie from .trkP, .trkPMD
      
      trkTS = obj.trkPTS;
      if isempty(trkTS)
        obj.xyPrdCurrMovie = [];
        obj.xyPrdCurrMovieIsInterp = [];
        return;
      end
            
      lObj = obj.lObj;
      movName = lObj.movieFilesAllFull{lObj.currMovie};
      [~,movS] = myfileparts(movName);
      nfrms = lObj.nframes;
      
      d = 2;
      nfids = obj.nPts;
      pTrk = obj.trkP(:,:,end);
      trkMD = obj.trkPMD;
      iPtTrk = obj.trkPiPt;
      nPtTrk = numel(iPtTrk);
      assert(isequal(size(pTrk),[size(trkMD,1) nPtTrk*d]));
      
      tfCurrMov = strcmp(trkMD.movS,movS); % these rows of trkMD are for the current Labeler movie
      nCurrMov = nnz(tfCurrMov);
      xyTrkCurrMov = reshape(pTrk(tfCurrMov,:)',nPtTrk,d,nCurrMov);
      
      frmCurrMov = trkMD.frm(tfCurrMov);
      xy = nan(nfids,d,nfrms);
      xy(iPtTrk,:,frmCurrMov) = xyTrkCurrMov;
      
      if obj.trkVizInterpolate
        [xy,isinterp3] = CPRLabelTracker.interpolateXY(xy);
        isinterp = CPRLabelTracker.collapseIsInterp(isinterp3);
      else
        isinterp = false(nfrms,1);
      end
      
      obj.xyPrdCurrMovie = xy;
      obj.xyPrdCurrMovieIsInterp = isinterp;
    end

    function vizHide(obj)
      [obj.hXYPrdRed.Visible] = deal('off');
    end
    
    function vizShow(obj)
      [obj.hXYPrdRed.Visible] = deal('on'); 
    end
    
    function vizInterpolateXYPrdCurrMovie(obj)
      [obj.xyPrdCurrMovie,isinterp3] = CPRLabelTracker.interpolateXY(obj.xyPrdCurrMovie);
      obj.xyPrdCurrMovieIsInterp = CPRLabelTracker.collapseIsInterp(isinterp3);
    end
    
  end

  %%
  methods (Static)
    
    function [xy,isinterp] = interpolateXY(xy)
      % xy (in): [npts d nfrm]
      %
      % xy (out): [npts d nfrm]. Like input, but with nans replaced with
      % values by interpolating along 3rd dim 
      % isinterp: [npts d nfrm] logical.
      
      isinterp = false(size(xy));
      for iPt = 1:size(xy,1)
        for d = 1:size(xy,2)
          z = squeeze(xy(iPt,d,:));
          tf = ~isnan(z);
          if any(tf) % must have at least one nonnan value to interpolate
            iGood = find(tf);
            iNan = find(~tf);
            z(iNan) = interp1(iGood,z(iGood),iNan);
            xy(iPt,d,:) = z;
            isinterp(iPt,d,iNan) = true;
          end
        end
      end
    end
    
    function isinterp1 = collapseIsInterp(isinterp3)
      % isinterp3: [nptsxdxnfrm] logical, see interpolateXY().
      %
      % isinterp1: [nfrm] logical. isinterp1(i) is true if all elements of
      % isinterp3(:,:,i) are true.
      
      [npts,d,nfrm] = size(isinterp3);
      isinterp2 = reshape(isinterp3,[npts*d,nfrm]);
      
      tfall = all(isinterp2,1);
      tfany = any(isinterp2,1);
      tfmixed = tfany & ~tfall;
      nfrmmixed = nnz(tfmixed);
      if nfrmmixed>0
        warning('CPRLabelTracker:isinterp','%d/%d frames have some interpolated and some non-interpolated tracking results. Treating these frames as non-interpolated.',...
          nfrmmixed,nfrm);
      end
      
      isinterp1 = tfall(:);
    end
    
    function tdPPJan(td,varargin)
      td.computeIpp([],[],[],'iTrl',1:td.N,'jan',true,varargin{:});
    end
    
  end
  
end