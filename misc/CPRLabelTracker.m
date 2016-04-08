classdef CPRLabelTracker < LabelTracker
  
  properties (Constant)
    SAVETOKEN_PROPS = {'trnDataTS' 'trnRes' 'trnResTS' 'trnResPallMD' ...
                       'trkP' 'trkPFull' 'trkPTS' 'trkPMD'};
  end
  
  properties
    % Training state -- set during .train()
    trnData % most recent training data
    trnDataTS % timestamp for trnData
    trnRes % most recent training results
    trnResTS % timestamp for trnRes
    trnResPallMD % movie/frame metadata for trnRes.pAll
    
    % Tracking state -- set during .track()
    trkP % [NTst D T+1] reduced/pruned tracked shapes
    trkPFull % [NTst RT D T+1] Tracked shapes full data
    trkPTS % timestamp for trkP*
    trkPMD % movie/frame md for trkP
    
    % View/presentation
    xyPrdCurrMovie; % [npts d nfrm] predicted labels for current Labeler movie
    hXYPrdRed; % [npts] plot handles for 'reduced' tracking results, current frame
  end
  properties (Dependent)
    nPts
  end
  
  methods
    function v = get.nPts(obj)
      v = obj.lObj.nLabelPoints;
    end
  end
  
  methods
    
    function obj = CPRLabelTracker(lObj)
      obj@LabelTracker(lObj);
    end
    
    function delete(obj)
      deleteValidHandles(obj.hXYPrdRed);
      obj.hXYPrdRed = [];
    end
    
  end
  
  methods
   
    function initHook(obj)
      obj.trnData = [];
      obj.trnDataTS = [];
      obj.trnRes = [];
      obj.trnResPallMD = [];
      obj.trnResTS = [];
      
      obj.trkP = [];
      obj.trkPFull = [];
      obj.trkPTS = [];
      obj.trkPMD = [];
            
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
    
    function td = prepareCPRData(obj,ppPrm,varargin)
      % td = prepareCPRData(obj,ppPrm,'all',varargin) % include all frames from all movies
      % td = prepareCPRData(obj,ppPrm,'lbl',varargin) % include all labeled frames from all movies
      % td = prepareCPRData(obj,ppPrm,iMovs,frms,varargin)
      %
      % Does not mutate obj
            
      if any(strcmp(varargin{1},{'all' 'lbl'}));
        type = varargin{1};
        varargin = varargin(2:end);
      else
        [iMovs,frms] = deal(varargin{1:2});
        varargin = varargin(3:end);
      end
      
      [useTrnH0,hWB] = myparse(varargin,...
        'useTDH0',false,... % if true, use trainData H0 for histEq (if histEq requested)
        'hWaitBar',[]);
      
      lObj = obj.lObj;
            
      if exist('type','var')>0
        td = CPRData(lObj.movieFilesAll,lObj.labeledpos,lObj.labeledpostag,...
          type,'hWaitBar',hWB);
      else
        td = CPRData(lObj.movieFilesAll,lObj.labeledpos,lObj.labeledpostag,...
          iMovs,frms,'hWaitBar',hWB);
      end
        
      md = td.MD;
      if ppPrm.histeq
        if useTrnH0
          H0 = obj.trnData.H0;
          assert(~isempty(H0));
        else
          H0 = [];
        end
        gHE = categorical(md.movS);
        td.histEq('g',gHE,'hWaitBar',hWB,'H0',H0);
      else
        fprintf(1,'Not doing histogram equalization.\n');
      end
      if ~isempty(ppPrm.channelsFcn)
        feval(ppPrm.channelsFcn,td,'hWaitBar',hWB);
      else
        fprintf(1,'Not computing channel features.');
      end
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
      
      hWB = waitbar(0);
      hTxt = findall(hWB,'type','text');
      hTxt.Interpreter = 'none';

      td = obj.prepareCPRData(prm.PreProc,'lbl','hWaitBar',hWB);
      td.iTrn = 1:td.N;
      td.summarize('movS',td.iTrn);
      
      obj.trnData = td;
      obj.trnDataTS = now;
      
      [Is,nChan] = td.getCombinedIs(td.iTrn);
      prm.Ftr.nChn = nChan;
      
      delete(hWB); % AL: get this guy in training?
      
      tr = train(td.pGTTrn,td.bboxesTrn,Is,...
          'modelPrms',prm.Model,...
          'regPrm',prm.Reg,...
          'ftrPrm',prm.Ftr,...
          'initPrm',prm.TrainInit,...
          'prunePrm',prm.Prune,...
          'docomperr',false,...
          'singleoutarg',true);
      obj.trnRes = tr;
      obj.trnResTS = now;
      obj.trnResPallMD = td.MD;
      
%       obj.loadXYPrdCurrMovie();
%       obj.newLabelerFrame();
    end
    
    function track(obj,iMovs,frms)
      if isempty(obj.trnRes)
        error('CPRLabelTracker:noRes','No tracker has been trained.');
      end
      
      prm = obj.readParamFile();

      hWB = waitbar(0);
      hTxt = findall(hWB,'type','text');
      hTxt.Interpreter = 'none';

      td = obj.prepareCPRData(prm.PreProc,iMovs,frms,'hWaitBar',hWB,'useTDH0',true);

      td.iTst = 1:td.N;
      td.summarize('movS',td.iTst);
%       obj.trnData = td;
%       obj.trnDataTS = now;

      delete(hWB);
 
      [Is,nChan] = td.getCombinedIs(td.iTst);
      prm.Ftr.nChn = nChan;
            
      %% Test on test set
      tr = obj.trnRes;
      prmInit = prm.TestInit;
      NTst = td.NTst;
      RT = prmInit.Nrep;
      Tp1 = tr.regModel.T+1;
      mdl = tr.regModel.model;
      
      pGTTrnNMu = nanmean(tr.regModel.pGtN,1);
      pIni = shapeGt('initTest',[],td.bboxesTst,mdl,[],...
        repmat(pGTTrnNMu,NTst,1),RT,prmInit.augrotate);
      VERBOSE = 0;
      [~,p_t] = rcprTest1(Is,tr.regModel,pIni,tr.regPrm,tr.ftrPrm,...
        td.bboxesTst,VERBOSE,tr.prunePrm);
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
      
      obj.trkP = pTstTRed;
      obj.trkPFull = pTstT;
      obj.trkPTS = now;
      obj.trkPMD = td.MD;
      
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
      
      if trkTS<obj.trnResTS || trkTS<obj.trnDataTS
        warning('CPRLabelTracker:trackOOD',...
          'Tracking results appear out-of-date.');
      end
        
      lObj = obj.lObj;
      movName = lObj.movieFilesAll{lObj.currMovie};
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
      s.trnDataMD = obj.trnData.MD;
      for p = obj.SAVETOKEN_PROPS, p=p{1}; %#ok<FXSET>
        s.(p) = obj.(p);
      end
    end
    
    function loadSaveToken(obj,s)
      assert(isequal(s.labelTrackerClass,class(obj)));
      if isempty(obj.trnData)
        % will remain empty; training results do not contain trnData
      else
        if ~isequaln(s.trnDataMD,obj.trnData.MD)
          % AL 20160404: will need to be updated soon
          error('CPRLabelTracker:loadSaveToken',...
            'Save training results based on different training data; aborting load.');
        end
      end
      
      for p = obj.SAVETOKEN_PROPS, p=p{1}; %#ok<FXSET>
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