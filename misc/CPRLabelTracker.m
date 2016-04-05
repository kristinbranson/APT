classdef CPRLabelTracker < LabelTracker
  
  properties (Constant)
    SAVETOKEN_PROPS = {'trnDataTS' 'trnRes' 'trnResMD' 'trnResTS'};
  end
  
  properties
    % core training state
    trnData % most recent training data
    trnDataTS % timestamp for trnData
    trnRes % most recent training results
    trnResMD % movie/frame metadata for trnRes
    trnResTS % timestamp for trnRes
    
    % for view/presentation
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
      obj.trnResMD = [];
      obj.trnResTS = [];
      obj.xyPrdCurrMovie = [];
      
      npts = obj.nPts;
      ptsClrs = obj.lObj.labelPointsPlotInfo.Colors;
      ax = obj.ax;
      cla(ax);
      hold(ax,'on');
      hTmp = gobjects(npts,1);
      for iPt = 1:npts
        clr = ptsClrs(iPt,:);
        hTmp(iPt) = plot(ax,nan,nan,'+','Color',clr);
      end
      
      deleteValidHandles(obj.hXYPrdRed);
      obj.hXYPrdRed = hTmp;      
    end
    
    function track(obj)      
      if isempty(obj.paramFile)
        error('CPRLabelTracker:noParams','Tracking parameter file needs to be set.');
      end
      prm = ReadYaml(obj.paramFile);
        
      lObj = obj.lObj;
      
      hWB = waitbar(0);
      hTxt = findall(hWB,'type','text');
      hTxt.Interpreter = 'none';
      
      % Create/preprocess the training data
      td = CPRData(lObj.movieFilesAll,lObj.labeledpos,lObj.labeledpostag,false,...
        'hWaitBar',hWB);
      md = td.MD;
      prmPP = prm.PreProc;
      if prmPP.histeq
        gHE = categorical(md.movS);
        td.histEq('g',gHE,'hWaitBar',hWB);
      else
        fprintf(1,'Not doing histogram equalization.');
      end
      if ~isempty(prmPP.channelsFcn)
        feval(prmPP.channelsFcn,td,'hWaitBar',hWB);
      else
        fprintf(1,'Not computing channel features.');
      end
      obj.trnData = td;
      obj.trnDataTS = now;
            
      td.iTrn = 1:td.N;
      td.summarize('movS',td.iTrn);

      [Is,nChan] = td.getTrnCombinedIs();
      prm.Ftr.nChn = nChan;
      
      delete(hWB);
      
      tr = train(td.pGTTrn,td.bboxesTrn,Is,...
          'modelPrms',prm.Model,...
          'regPrm',prm.Reg,...
          'ftrPrm',prm.Ftr,...
          'initPrm',prm.Init,...
          'prunePrm',prm.Prune,...
          'docomperr',false,...
          'singleoutarg',true);
      obj.trnRes = tr;
      obj.trnResTS = now;
      obj.trnResMD = td.MD;
      
      obj.loadXYPrdCurrMovie();
      obj.newLabelerFrame();
    end
    
    function loadXYPrdCurrMovie(obj)
      % sets .xyPrdCurrMovie from tracking results for current movie
      
      lObj = obj.lObj;
      movName = lObj.movieFilesAll{lObj.currMovie};
      nfrms = lObj.nframes;
      
      tr = obj.trnRes;
      trMD = obj.trnResMD;
      mdl = tr.regModel.model;
      pTrk = tr.pAll(:,:,end);
      assert(isequal(size(pTrk),[size(trMD,1) mdl.D]));
      
      xy = nan(mdl.nfids,mdl.d,nfrms);
      tfCurrMov = strcmp(trMD.mov,movName); % these rows of trnData/MD are for the current Labeler movie
      nCurrMov = nnz(tfCurrMov);
      xyTrkCurrMov = reshape(pTrk(tfCurrMov,:)',mdl.nfids,mdl.d,nCurrMov); % [npt x d x nCurrMov]
      
      frmCurrMov = trMD.frm(tfCurrMov);
      xy(:,:,frmCurrMov) = xyTrkCurrMov;
      obj.xyPrdCurrMovie = xy;
    end
      
    function newLabelerFrame(obj)
      if isempty(obj.trnRes)
        return;
      end
      
      frm = obj.lObj.currFrame;
      xy = obj.xyPrdCurrMovie(:,:,frm); % [npt x d]
      npts = obj.nPts;
      hXY = obj.hXYPrdRed;
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