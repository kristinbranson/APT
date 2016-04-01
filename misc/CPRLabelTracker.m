classdef CPRLabelTracker < LabelTracker
  
  properties
    doHE % logical scalar
    preProcFcn % function handle to preprocessor; caled as ppFcn(obj.td)
    trnPrmFile % training parameters file    
    
    trnData % most recent training data
    trnDataTS % timestamp for trnData
    trnRes % most recent training results
    
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
      obj.initPredVis();
    end
    
    function delete(obj)
      deleteValidHandles(obj.hXYPrdRed);
    end
    
  end
  
  methods
    
    function track(obj)
      lObj = obj.lObj;
      
      hWB = waitbar(0);
      hTxt = findall(hWB,'type','text');
      hTxt.Interpreter = 'none';
      
      % Create/preprocess the training data
      td = CPRData(lObj.movieFilesAll,lObj.labeledpos,lObj.labeledpostag,false,...
        'hWaitBar',hWB);
      md = td.MD;
      if obj.doHE
        gHE = categorical(md.movS);
        td.histEq('g',gHE,'hWaitBar',hWB);
      end
      if ~isempty(obj.preProcFcn)
        obj.preProcFcn(td,'hWaitBar',hWB);
      end
      obj.trnData = td;
      obj.trnDataTS = now;
      
      % Read the training parameters
      tp = ReadYaml(obj.trnPrmFile);
      
      td.iTrn = 1:td.N;
      td.summarize('movS',td.iTrn);

      [Is,nChan] = td.getTrnCombinedIs();
      tp.Ftr.nChn = nChan;
      
      delete(hWB);
      
      tr = train(td.pGTTrn,td.bboxesTrn,Is,...
          'modelPrms',tp.Model,...
          'regPrm',tp.Reg,...
          'ftrPrm',tp.Ftr,...
          'initPrm',tp.Init,...
          'prunePrm',tp.Prune,...
          'docomperr',false,...
          'singleoutarg',true);
      obj.trnRes = tr;
      
      obj.loadXYPrdCurrMovie();
      obj.newLabelerFrame();
    end
    
    function loadXYPrdCurrMovie(obj)
      % sets .xyPrdCurrMovie from tracking results for current movie
      
      lObj = obj.lObj;
      movName = lObj.movieFilesAll{lObj.currMovie};
      nfrms = lObj.nframes;
      
      td = obj.trnData;
      pTrk = obj.trnRes.pAll(:,:,end);
      assert(isequal(size(pTrk),size(td.pGT)));
      
      xy = nan(td.nfids,td.d,nfrms);
      trnMD = td.MD;
      tfCurrMov = strcmp(trnMD.mov,movName); % these rows of trnData/MD are for the current Labeler movie
      nCurrMov = nnz(tfCurrMov);
      xyTrkCurrMov = reshape(pTrk(tfCurrMov,:)',td.nfids,td.d,nCurrMov); % [npt x d x nCurrMov]
      
      frmCurrMov = trnMD.frm(tfCurrMov);
      xy(:,:,frmCurrMov) = xyTrkCurrMov;
      obj.xyPrdCurrMovie = xy;
    end
    
    function initPredVis(obj)
      % Currently just inits .hXYPrdRed
            
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
    
  end
  
  methods (Static)
    function tdPPJan(td,varargin)
      td.computeIpp([],[],[],'iTrl',1:td.N,'jan',true,varargin{:});
    end
  end
  
end