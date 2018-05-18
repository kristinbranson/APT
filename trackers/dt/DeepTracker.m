classdef DeepTracker < LabelTracker
  
  properties
    sPrm % new-style DT params
    
    % bg trn monitor
    bgTrnMonitorClient % BGClient obj
    bgTrnMonitorWorkerObj; % scalar "detached" object that is deep-copied onto 
      % workers. Note, this is not the BGWorker obj itself
    bgTrnMonitorResultsMonitor % object with resultsreceived() method 
    
  end
  properties (Dependent)
    bgReady % If true, asyncPrepare() has been called and asyncStartBGWorker() can be called
  end

  properties
    % track res
    trkP % [NTst trkD] tracked shapes. In ABSOLUTE coords    
    trkPTS % [NTstx1] timestamp for trkP*
    trkPMD % [NTst <ncols>] table. cols: .mov, .frm, .iTgt
           % .mov has class movieIndex
           
    % viz
    xyPrdCurrMovie; % [npts d nfrm ntgt] predicted labels for current Labeler movie
    hXYPrdRed; % [npts] plot handles for 'reduced' tracking results, current frame and target
    hXYPrdRedOther; % [npts] plot handles for 'reduced' tracking results, current frame, non-current-target
    xyVizPlotArgs; % cell array of args for regular tracking viz    
    xyVizPlotArgsNonTarget; % " for non current target viz
  end
  properties (Dependent)
    nPts % number of label points     
    %hasTrained
  end
  
  events
    % Thrown when trkP/trkPMD are mutated (basically)
    newTrackingResults 
  end
  
  methods
    function v = get.nPts(obj)
      v = obj.lObj.nLabelPoints;
    end
    function v = get.bgReady(obj)
      v = ~isempty(obj.bgTrnMonitorClient);
    end
  end
  
  methods
    function obj = DeepTracker(lObj)
      obj@LabelTracker(lObj);
    end    
    function initHook(obj)
      obj.bgReset();
      obj.trackResInit();
      obj.vizInit();
    end
  end
  
  methods
    
    function train(obj)
      % Caller: make sure project is saved. make sure proj has a name, eg
      % from projAssign...

      lblObj = obj.lObj;     
      projname = lblObj.projname;
      assert(~isempty(projname));      
      jobID = sprintf('%s_%s',projname,datestr(now,'yyyymmddTHHMMSS'));
      
      % Write stripped lblfile to cacheDir
      s = lblObj.trackCreateStrippedLbl();
      cacheDir = obj.sPrm.CacheDir;
      dlLblFile = fullfile(cacheDir,[jobID '.lbl']);
      save(dlLblFile,'-mat','-struct','s');
      fprintf('Saved stripped lbl file: %s\n',dlLblFile);

      % start training
      aptintrf = fullfile(obj.posetfroot,'APT_interface.py');
      cmd = sprintf('%s -n %s %s train',aptintrf,obj.jobID,obj.dlLblFile);
      fprintf(1,'Running %s\n',cmd);
            
      % call BG Train Monitor
      obj.bgPrepareTrainMonitor(dlLblFile,jobID);
      obj. bgStartTrainMonitor();
    end
    
  end
  
  %% BG Train Monitor
  methods
    
    function bgReset(obj) %,tfwarn)
      % Clear all async* state
      %
      % See asyncDetachedCopy for the state copied onto the BG worker. The
      % BG worker depends on: .sPrm, preprocessing parameters/H0/etc for
      % .data*, .trnRes*.
      %
      % - Note, when you change eg params, u need to call this. etc etc.
      % Any mutation that alters PP, train/track on the BG worker...
      % - .trnRes* state is set during train/retrain operations. At the
      % start of these operations we do an asyncReset();
      % - trackResInit() is sometimes called when a change in prune 
      % parameters etc is made. In these cases an asyncReset() will follow

%       if exist('tfwarn','var')==0
%         tfwarn = false;
%       end
      
%       obj.asyncPredictOn = false;
      if ~isempty(obj.bgTrnMonitorClient)
        delete(obj.bgTrnMonitorClient);
%       else
%         tfwarn = false;
      end
      obj.bgTrnMonitorClient = [];
      
      if ~isempty(obj.bgTrnMonitorWorkerObj)
        delete(obj.bgTrnMonitorWorkerObj)
      end
      obj.bgTrnMonitorWorkerObj = [];
      
      if ~isempty(obj.bgTrnMonitorResultsMonitor)
        delete(obj.bgTrnMonitorResultsMonitor);
      end
      obj.bgTrnMonitorResultsMonitor = [];
      
%       if tfwarn
%         warningNoTrace('CPRLabelTracker:bg','Cleared background tracker.');
%       end
    end
    
    function bgPrepareTrainMonitor(obj,dlLblFile,jobID)
      obj.bgReset();

      objMon = DeepTrackerTrainingMonitor(obj.lObj.nview);
      cbkResult = @objMon.resultsReceived;
      workerObj = DeepTrackerTrainingWorkerObj(dlLblFile,jobID);
      bgc = BGClient;
      fprintf(1,'Configuring background worker...\n');
      bgc.configure(cbkResult,workerObj,'compute');
      obj.bgTrnMonitorClient = bgc;
      obj.bgTrnMonitorWorkerObj = workerObj;
      obj.bgTrnMonitorResultsMonitor = objMon;
    end
    
    function bgStartTrainMonitor(obj)
      assert(obj.bgReady);
      obj.bgTrnMonitorClient.startWorker('workerContinuous',true,...
        'continuousCallInterval',10);
    end

    function bgStopTrainMonitor(obj)
      obj.bgTrnMonitorClient.stopWorker();
    end
        
%       switch sRes.action
%         case 'track'
%           obj.updateTrackRes(res.trkPMDnew,res.pTstTRed,res.pTstT);
%           obj.vizLoadXYPrdCurrMovieTarget();
%           obj.newLabelerFrame();
%           notify(obj,'newTrackingResults');
%         case BGWorker.STATACTION
%           computeTimes = res;
%           CPRLabelTracker.asyncComputeStatsStc(computeTimes);
    
  end
  
  methods
    
    function [trkPMDnew,pTstTRed,pTstT] = trackCore(obj,tblP)
      prm = obj.sPrm;
      if isempty(prm)
        error('CPRLabelTracker:param','Please specify tracking parameters.');
      end
      if ~all([obj.trnResRC.hasTrained])
        error('CPRLabelTracker:track','No tracker has been trained.');
      end
                            
      %%% data
      [d,dataIdx] = obj.lObj.preProcDataFetch(tblP);
      d.iTst = dataIdx;
      fprintf(1,'Track data summary:\n');
      d.summarize('mov',d.iTst);
                
      [Is,nChan] = d.getCombinedIs(d.iTst);
      prm.Ftr.nChn = nChan;
        
      %% Test on test set; fill/generate pTstT/pTstTRed for this chunk
      NTst = d.NTst;
      RT = prm.TestInit.Nrep;
      nview = obj.sPrm.Model.nviews;
      nfids = prm.Model.nfids;
      assert(nview==numel(obj.trnResRC));
      assert(nview==size(Is,2));
      assert(prm.Model.d==2);
      Dfull = nfids*nview*prm.Model.d;
      pTstT = nan(NTst,RT,Dfull,prm.Reg.T+1);
      pTstTRed = nan(NTst,Dfull);
      pTstTPruneMD = array2table(nan(NTst,0));
      for iView=1:nview % obj CONST over this loop
        rc = obj.trnResRC(iView);
        IsVw = Is(:,iView);
        bboxesVw = CPRData.getBboxes2D(IsVw);
        if nview==1
          assert(isequal(bboxesVw,d.bboxesTst));
        end
          
        % Future todo, orientationThetas
        % Should break internally if 'orientationThetas' is req'd
        [p_t,pIidx,p0,p0Info] = rc.propagateRandInit(IsVw,bboxesVw,...
          prm.TestInit);

        trkMdl = rc.prmModel;
        trkD = trkMdl.D;
        Tp1 = rc.nMajor+1;
        pTstTVw = reshape(p_t,[NTst RT trkD Tp1]);
        
        %% Prune
        [pTstTRedVw,pruneMD] = CPRLabelTracker.applyPruning(...
          pTstTVw(:,:,:,end),d.MDTst(:,MFTable.FLDSID),prm.Prune);
        szassert(pTstTRedVw,[NTst trkD]);
        if nview>1
          pruneMD = tblfldsmodify(pruneMD,@(x)[x '_vw' num2str(iView)]);
        end
        pTstTPruneMD = [pTstTPruneMD pruneMD]; %#ok<AGROW>        
        
        assert(trkD==Dfull/nview);
        assert(mod(trkD,2)==0);
        iFull = (1:nfids)+(iView-1)*nfids;
        iFull = [iFull,iFull+nfids*nview]; %#ok<AGROW>
        pTstT(:,:,iFull,:) = pTstTVw;
        pTstTRed(:,iFull) = pTstTRedVw;
      end % end obj CONST
        
      fldsTmp = MFTable.FLDSID;
      if any(strcmp(d.MDTst.Properties.VariableNames,'roi'))
        fldsTmp{1,end+1} = 'roi';
      end
      trkPMDnew = d.MDTst(:,fldsTmp);
      trkPMDnew = [trkPMDnew pTstTPruneMD];
      obj.updateTrackRes(trkPMDnew,pTstTRed,pTstT);
    end

    
    function trackResInit(obj)
      obj.trkP = [];
      obj.trkPTS = zeros(0,1);
      obj.trkPMD = MFTable.emptyTable(MFTable.FLDSID);
    end
    function vizInit(obj)
      obj.xyPrdCurrMovie = [];
      deleteValidHandles(obj.hXYPrdRed);
      obj.hXYPrdRed = [];
      deleteValidHandles(obj.hXYPrdRedOther);
      obj.hXYPrdRedOther = [];
      
      % init .xyVizPlotArgs*
      trackPrefs = obj.lObj.projPrefs.Track;
      plotPrefs = trackPrefs.PredictPointsPlot;
      plotPrefs.PickableParts = 'none';
      obj.xyVizPlotArgs = struct2paramscell(plotPrefs);
      obj.xyVizPlotArgsNonTarget = obj.xyVizPlotArgs; % TODO: customize
      
      npts = obj.nPts;
      ptsClrs = obj.lObj.labelPointsPlotInfo.Colors;
      ax = obj.ax;
      %arrayfun(@cla,ax);
      arrayfun(@(x)hold(x,'on'),ax);
      ipt2View = obj.lObj.labeledposIPt2View;
      hTmp = gobjects(npts,1);
      hTmpOther = gobjects(npts,1);
      hTmp2 = gobjects(npts,1);
      for iPt = 1:npts
        clr = ptsClrs(iPt,:);
        iVw = ipt2View(iPt);
        hTmp(iPt) = plot(ax(iVw),nan,nan,obj.xyVizPlotArgs{:},'Color',clr);
        hTmpOther(iPt) = plot(ax(iVw),nan,nan,obj.xyVizPlotArgs{:},'Color',clr);        
        hTmp2(iPt) = scatter(ax(iVw),nan,nan);
        setIgnoreUnknown(hTmp2(iPt),'MarkerFaceColor',clr,...
          'MarkerEdgeColor',clr,'PickableParts','none',...
          obj.xyVizFullPlotArgs{:});
      end
      obj.hXYPrdRed = hTmp;
      obj.hXYPrdRedOther = hTmpOther;
      obj.setHideViz(obj.hideViz);
    end
    
  end
  
  %%
  methods
    function setParams(obj,sPrm)
      % XXX: invalidating/clearing state
      obj.sPrm = sPrm;
    end
    function sPrm = getParams(obj)
      sPrm = obj.sPrm;
    end
    function s = getSaveToken(obj)
      s = struct();
      s.sPrm = obj.sPrm;
    end
    function loadSaveToken(obj,s)
      obj.sPrm = s.sPrm;
    end
  end
  
end