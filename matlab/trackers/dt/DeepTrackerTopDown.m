classdef DeepTrackerTopDown < DeepTracker
  % Top-down/two-stage DeepTracker
  %
  % Inheriting here but composing another option. (Lazy to write fwding 
  % methods etc). The base/superclass deeptracker state is for the stage2
  % tracker, as in multianimal-with-trx tracking. DeepTrackerTopDown adds 
  % an additional DeepTracker object to represent detection/stage1.
  
  % Reqs/factors
  % trnpack: just want to gen 1. AND, put it in a folder.
  % monitor: prob best to just have one. make new trn/trk vizers
  % ideally, can retrain each stage separately.
  %   if multple GPUs, spawn both stages. else, run serially. 
  % track. needs to run serially.
  % trackres. tracking results/viz interesting for both stages potentially.
  
  % iter 2 notes
  % what's in DeepTracker state?
  % - Metadata: nview, lObj, nPts. These can be shared between stg1/2.
  % - Algo stuff: isHeadTail, algoName, trnNetType/Mode. need separate.
  % - Params: shared, as stg1/detect params live in their own place.
  % - codegen/be stuff. mostly shared, but maybe a couple separate.
  % - DMC/trnres. could be nonscalar+shared, or indep. note how multiview
  %   is handled with nonscalar DMC in a single DT.
  % - trnMOn. prob one is ideal that looks over two DMCs.
  % - trkmon. prob one is ideal.
  % -   ** just make new Viz classes for trn/trkmon!
  % - trkRes. need separate.
  % - trkViz. need separate.
  
  % It looks like the TrackDB/TrackViz could be factored out which we
  % already suspected.
  % TrackDB: container mapping movies->TrkFiles. TrkFiles already have a
  %  fetch API, but the DB needs to handle merging multiple TrkFiles etc.
  % TrackVizer. This is listeners + trkViz. It's already kind of factored
  % out.
  % So, for .stage1, we will use the DT object just for its
  % TrackDB/TrackVizer functionalities. Everything else will be handled in
  % this object

  % iter 1 notes
  % could we somehow use two without mucking up DeepTracker too mcuh?
  % - OK trnpack. trnspawn, allow specification of pre-gened tp.
  % - monitor. support indep retrain of stages. i guess try having 2
  % monitors not horrible. 
  % - params. just force a copy not horrible.
  % - GPUs: assume two or JRC first, so no GPU shortage.
  % - track: trkcomplete trigger.
  
  properties
    forceSerial = false;
    
    stage1Tracker % See notes above
    
    trnDoneStage1 % logical
    trnDoneStage2 % logical
  end
  properties (Dependent)
    isHeadTail
    topDownTypeStr
  end
  methods
    function v = getAlgorithmNameHook(obj)
      v = sprintf('MA Top Down');%,obj.trnNetMode.shortCode,...
%        obj.stage1Tracker.trnNetMode.shortCode);
    end
    function v = getAlgorithmNamePrettyHook(obj)
      v = sprintf('Top-Down (%s): %s + %s',obj.topDownTypeStr,...
        obj.stage1Tracker.trnNetType.displayString,...
        obj.trnNetType.displayString);
    end
    function v = getNetsUsed(obj)
      v = cellstr([obj.stage1Tracker.trnNetType; obj.trnNetType]);
    end
    function v = getNumStages(obj)
      v = 2;
    end
    function v = get.isHeadTail(obj)
      v = obj.trnNetMode.isHeadTail;
    end
    function v = get.topDownTypeStr(obj)
      if obj.isHeadTail
        v = 'head/tail';
      else
        v = 'bbox';
      end
    end
  end
    
  methods 

    function obj = DeepTrackerTopDown(lObj,stg1ctorargs,stg2ctorargs)
      obj@DeepTracker(lObj,stg2ctorargs{:});
      dt = DeepTracker(lObj,stg1ctorargs{:});
      obj.stage1Tracker = dt;      
    end
    
    function initHook(obj)
      initHook@DeepTracker(obj);
      obj.stage1Tracker.initHook();      
    end
    
    function setAllParams(obj,sPrmAll)
      obj.stage1Tracker.setAllParams(sPrmAll);
      setAllParams@DeepTracker(obj,sPrmAll);      
    end
    
    function sloc = genStrippedLblTrnPack(obj,dlLblFileLcl)
      % Generate/write a trnpack/stripped lbl; can be used for both stages.
      %
      
      [dlLblFileLclDir,slblf,slble] = fileparts(dlLblFileLcl);
      if exist(dlLblFileLclDir,'dir')==0
        fprintf('Creating dir: %s\n',dlLblFileLclDir);
        [succ,msg] = mkdir(dlLblFileLclDir);
        if ~succ
          error('Failed to create dir %s: %s',dlLblFileLclDir,msg);
        end
      end
      
      packdir = dlLblFileLclDir;
      [~,~,sloc,~] = Lbl.genWriteTrnPack(obj.lObj,packdir,...
        'strippedlblname',[slblf slble]);
    end
    
    function tf = isTrkFiles(obj)
      tf = isTrkFiles@DeepTracker(obj) || obj.stage1Tracker.isTrkFiles();
    end
    
    function trnSpawnBsubDocker(obj,backEnd,trnType,modelChainID,varargin)
      [wbObj,prev_models] = myparse(varargin,...        
        'wbObj',[],'prev_models',[] ...
        );
            
      cacheDir = obj.lObj.DLCacheDir;      
      % Currently, cacheDir must be visible on the JRC shared filesys.
      % In the future, we may need i) "localWSCache" and ii) "jrcCache".
     
%       nvw = obj.lObj.nview;
%      isSerialTrain = false;
      % backend; implement getFreeGPUs for bsub
      if obj.forceSerial
        nTrainJobs = 1;
        warningNoTrace('Forcing serial train.');
      else
        nTrainJobs = 2;
      end
      if backEnd.type == DLBackEnd.Docker || backEnd.type == DLBackEnd.Conda,
        gpuids = backEnd.getFreeGPUs(nTrainJobs);
        if numel(gpuids) < nTrainJobs
          if nTrainJobs == 1 || numel(gpuids)<1
            error('No GPUs with sufficient RAM available locally');
          else
            gpuids = gpuids(1);
            %           isSerialTrain = true;
            nTrainJobs = 1;
          end
        else
          gpuids = gpuids(1:nTrainJobs);
        end
      end
      
       % Base DMC, to be further copied/specified 
      objStg1 = obj.stage1Tracker;
      sPrmGD = obj.sPrmAll.ROOT.DeepTrack.GradientDescent;
      sPrmGDStg1 = obj.sPrmAll.ROOT.MultiAnimal.Detect.DeepTrack.GradientDescent;
      dmc = DeepModelChainOnDisk(... 
        'rootDir',cacheDir,...
        'projID',obj.lObj.projname,...
        'netType',char(objStg1.trnNetType),...
        'netMode',objStg1.trnNetMode,...
        'view',0,... % to be filled in 
        'modelChainID',modelChainID,...
        'trainID','',... % to be filled in 
        'trainType',trnType,...
        'iterFinal',sPrmGDStg1.dl_steps, ...
        'isMultiView',false,...
        'reader',DeepModelChainReader.createFromBackEnd(backEnd),...
        'prev_models',prev_models,...
        'filesep',obj.filesep...
        );
      dmc(2) = dmc(1).copy();
      dmc(2).netType = char(obj.trnNetType);
      dmc(2).netMode = obj.trnNetMode;
      dmc(2).iterFinal = sPrmGD.dl_steps;

      switch backEnd.type
        case DLBackEnd.Bsub
          aptroot = backEnd.bsubSetRootUpdateRepo(cacheDir);
        case {DLBackEnd.Conda,DLBackEnd.Docker},
          aptroot = APT.Root;
          obj.downloadPretrainedWeights('aptroot',aptroot); 
      end
      
      % create/ensure stripped lbl; set trainID
      tfGenNewStrippedLbl = trnType==DLTrainType.New || ...
                            trnType==DLTrainType.RestartAug;
      
      trnCmdType = trnType;
      
      netObj = obj.trnNetType;
      if false % ~isempty(existingTrnPackSLbl)        
        dlLblFileLcl = existingTrnPackSLbl;
        [tpdir,dllblf,~] = fileparts(dlLblFileLcl);

        % dlLblFileLcl should look like <modelChainID>_<trainID>.lbl
        pat = sprintf('%s_(?<trainID>[0-9T]+)$',modelChainID);
        toks = regexp(dllblf,pat,'names');        
        trainID = toks.trainID;
        dmc.trainID = trainID;
        assert(strcmp(dmc.lblStrippedLnx,dlLblFileLcl));
        
        tpjson = fullfile(tpdir,'trnpack.json');
        tp = Lbl.hlpLoadJson(tpjson);
        nlbls = arrayfun(@(x)size(x.p,2),tp);
        dmc.nLabels = nlbls;
        
        fprintf('Using pre-existing stripped lbl/trnpack: %s.\n',tpdir);
        fprintf('trainID: %s. nLabels: %d.\n',trainID,dmc.nLabels);
        
      elseif tfGenNewStrippedLbl
        trainID = datestr(now,'yyyymmddTHHMMSS');
        % Note dmc.trainID used in eg lblStrippedLnx
        [dmc.trainID] = deal(trainID);

        dlLblFileLcl = dmc(1).lblStrippedLnx;
        tfRequiresTrnPack = netObj.requiresTrnPack(obj.trnNetMode);
        assert(tfRequiresTrnPack);
        sloc = obj.genStrippedLblTrnPack(dlLblFileLcl);
        nlbls = numel(sloc);
        [dmc.nLabels] = deal(nlbls);

      else % Restart
        assert(false,'Restarts unsupported for multianimal trackers.');        
      end

      % At this point
      % We have (modelChainID,trainID). stripped lbl is on disk. 

      %syscmds = cell(nTrainJobs,1);
      
      switch backEnd.type
        case DLBackEnd.Bsub
          mntPaths = obj.genContainerMountPathBsubDocker(backEnd);
          singargs = {'bindpath',mntPaths};
          bsubargs = {'gpuqueue' obj.jrcgpuqueue 'nslots' obj.jrcnslots};          
          tfSerial = (nTrainJobs==1);
          syscmds = DeepTrackerTopDown.tdTrainCodeGenSSHBsubSingDMC(...
            tfSerial,aptroot,dmc,trnCmdType,bsubargs,singargs);
          assert(numel(syscmds)==nTrainJobs);
          
        case DLBackEnd.Docker
          mntPaths = obj.genContainerMountPathBsubDocker(backEnd);
          tfSerial = (nTrainJobs==1);
          [syscmds,containerNames] = ...
            DeepTrackerTopDown.tdTrainCodeGenDockerDMC(tfSerial,...
            backEnd,dmc,trnCmdType,mntPaths,gpuids);
          logfiles = {dmc.trainLogLnx}';
          logfiles = logfiles(1:numel(syscmds)); % for serial, use first
          logcmds = cellfun( ...
            @(zcntnr,zlogfile) sprintf('%s logs -f %s &> "%s" &',...
                                      backEnd.dockercmd,zcntnr,zlogfile),...
              containerNames(:),logfiles(:),'uni',0);
          
        case DLBackEnd.Conda
          assert(false,'Unsupported'); % XXX TODO
          condaargs = {'condaEnv',obj.condaEnv};
          for ivw=1:nvw,
            if ivw>1
              dmc(ivw) = dmc(1).copy();
            end
            dmc(ivw).view = ivw-1; % 0-based
            if ivw <= nTrainJobs,
              gpuid = gpuids(ivw);
              syscmds{ivw} = ...
                DeepTracker.trainCodeGenCondaDMC(dmc(ivw),gpuid,...
                'isMultiView',isSerialTrain,'trnCmdType',trnCmdType,...
                'condaargs',condaargs);
            end
          end
        otherwise
          assert(false);
      end
      
      if obj.dryRunOnly
        cellfun(@(x)fprintf(1,'Dry run, not training: %s\n',x),syscmds);
      else        
        %TRNMON = 'TrkTrnMonVizSimpleStore';
        %fprintf(2,'hardcode trnmon: %s\n',TRNMON);
        obj.bgTrnStart(backEnd,dmc,'trnVizArgs',{'nsets',2}); 
              % 'trnStartCbk',trnStartCbk,...
                                     % 'trnCompleteCbk',trnCompleteCbk);
        
        bgTrnWorkerObj = obj.bgTrnMonBGWorkerObj;
        
        % spawn training
        if backEnd.type==DLBackEnd.Docker
          bgTrnWorkerObj.jobID = cell(1,nTrainJobs);
          for iview=1:nTrainJobs
            fprintf(1,'%s\n',syscmds{iview});
            [st,res] = system(syscmds{iview});
            if st==0
              bgTrnWorkerObj.parseJobID(res,iview);
              
              fprintf(1,'%s\n',logcmds{iview});
              [st2,res2] = system(logcmds{iview});
              if st2==0
              else
                fprintf(2,'Failed to spawn logging job for view %d: %s.\n\n',...
                  iview,res2);
              end
            else
              fprintf(2,'Failed to spawn training job for view %d: %s.\n\n',...
                iview,res);
            end            
          end
        elseif backEnd.type==DLBackEnd.Conda
          bgTrnWorkerObj.jobID = cell(1,nTrainJobs);
          for iview=1:nTrainJobs
            fprintf(1,'%s\n',syscmds{iview});
            [job,st,res] = parfevalsystem(syscmds{iview});
            if ~st,
              bgTrnWorkerObj.parseJobID(job,iview);
            else
              fprintf(2,'Failed to spawn training job for view %d: %s.\n\n',...
                iview,res);
            end            
          end
        else
          bgTrnWorkerObj.jobID = nan(1,nTrainJobs);
          %assert(nTrainJobs==numel(dmc));
          for iview=1:nTrainJobs
            syscmdrun = syscmds{iview};
            fprintf(1,'%s\n',syscmdrun);
            
            cmdfile = dmc(iview).cmdfileLnx;
            %assert(exist(cmdfile,'file')==0,'Command file ''%s'' exists.',cmdfile);
            [fh,msg] = fopen(cmdfile,'w');
            if isequal(fh,-1)
              warningNoTrace('Could not open command file ''%s'': %s',cmdfile,msg);
            else
              fprintf(fh,'%s\n',syscmdrun);
              fclose(fh);
              fprintf(1,'Wrote command to cmdfile %s.\n',cmdfile);
            end
            
            [st,res] = system(syscmdrun);
            if st==0
              PAT = 'Job <(?<jobid>[0-9]+)>';
              stoks = regexp(res,PAT,'names');
              if ~isempty(stoks)
                jobid = str2double(stoks.jobid);
              else
                jobid = nan;
                warningNoTrace('Failed to ascertain jobID.');
              end
              fprintf('Training job (view %d) spawned, jobid=%d.\n\n',...
                iview,jobid);
              % assigning to 'local' workerobj, not the one copied to workers
              bgTrnWorkerObj.jobID(iview) = jobid;
            else
              fprintf('Failed to spawn training job for view %d: %s.\n\n',...
                iview,res);
            end
          end
        end        
        obj.trnLastDMC = dmc;
      end
    end
    
    function trnSpawnAWS(varargin)
      assert(false,'Unsupported');
    end    
    
    function  [tfCanTrain,reason] = canTrain(obj)
      [tfCanTrain,reason] = canTrain@DeepTracker(obj);
      if ~tfCanTrain
        return;
      end
      if obj.isHeadTail && (isempty(obj.lObj.skelHead) || isempty(obj.lObj.skelTail))
        tfCanTrain = false;
        reason = 'For head/tail tracking, please specify head and tail landmarks under Track > Landmark Parameters';
        return;
      end
      
      tfCanTrain = true;
      reason = '';
    end
      
%       obj1 = obj2.stage1Tracker;
%       objs = {obj1 obj2};
%       for stg=1:2
%         if objs{stg}.bgTrnIsRunning
%           error('Stage %d training is already in progress.',stg);
%         end
%         if objs{stg}.bgTrkIsRunning
%           error('Stage %d tracking is in progress.',stg);
%         end
%       end
%             
%       lblObj = obj2.lObj;
%       projname = lblObj.projname;
%       if isempty(projname)
%         error('Please give your project a name. The project name will be used to identify your trained models on disk.');
%       end
%       
%       trnBackEnd = lblObj.trackDLBackEnd;
%       fprintf('Top-down multianimal tracking\n');
%       fprintf('Your stage 1 (detection) deep net type is: %s\n',char(obj1.trnNetType));
%       fprintf('Your stage 2 (pose tracking) deep net type is: %s\n',char(obj2.trnNetType));
%       fprintf('Your training backend is: %s\n',char(trnBackEnd.type));
%       %fprintf('Your training vizualizer is: %s\n',obj2.bgTrnMonitorVizClass);
%       fprintf(1,'\n');
      
%       for stg=1:2
%         o = objs{stg};
%         if o.isTrkFiles(),
%           if isempty(o.skip_dlgs) || ~o.skip_dlgs
%             qstr = sprintf('Stage %d: Tracking results exist for previous deep trackers. When training stops, these will be deleted. Continue training?',stg);
%             res = questdlg(qstr,'Continue training?','Yes','No','Cancel','Yes');
%             if ~strcmpi(res,'Yes'),
%               return;
%             end
%           end
%         end
% 
%         o.setAllParams(lblObj.trackGetParams());
% 
%         if isempty(o.sPrmAll)
%           error('No tracking parameters have been set.');
%         end
% 
%         o.bgTrnReset();
% %         if ~isempty(oldVizObj),
% %           delete(oldVizObj);
% %         end
%       end
      
%       modelChain0 = obj2.trnName;
%       dlTrnType = DLTrainType.New;
%       switch dlTrnType
%         case DLTrainType.New
%           modelChain = datestr(now,'yyyymmddTHHMMSS');
%           if ~isempty(modelChain0)
%             assert(~strcmp(modelChain,modelChain0));
%             fprintf('Training new model %s.\n',modelChain);
%           end
% %         case {DLTrainType.Restart DLTrainType.RestartAug}
% %           if isempty(modelChain0)
% %             error('Model has not been trained.');
% %           end
% %           modelChain = modelChain0;
% %           fprintf('Restarting train on model %s.\n',modelChain);
%         otherwise
%           assert(false);
%       end
%             
%       trainID = modelChain;
%       dmcDummy = DeepModelChainOnDisk(...
%         'rootDir',obj2.lObj.DLCacheDir,...
%         'projID',obj2.lObj.projname,...
%         'modelChainID',modelChain,...
%         'trainID',trainID ...
%         );
%       dlLblFileLcl = dmcDummy.lblStrippedLnx;
%       obj2.genStrippedLblTrnPack(dlLblFileLcl);
%       switch trnBackEnd.type        
%         case {DLBackEnd.Bsub DLBackEnd.Conda DLBackEnd.Docker}          
%           
%           obj2.trnDoneStage1 = false;
%           obj2.trnDoneStage2 = false;
%           
%           trainStartCbk1 = @(s,e)0; % "do nothing"
%           % trainStartCbk2 = []; % No need to supply, use default
%           trainCompleteCbk1 = @(s,e)obj2.trainCompleteCbkStg1(s,e);
%           trainCompleteCbk2 = @(s,e)obj2.trainCompleteCbkStg2(s,e);                    
%           args = {'wbObj' wbObj 'existingTrnPackSLbl',dlLblFileLcl};
%           obj1.trnSpawnBsubDocker(trnBackEnd,dlTrnType,modelChain,...
%             'trnStartCbk',trainStartCbk1,...
%             'trnCompleteCbk',trainCompleteCbk1,...
%             args{:});
%           obj2.trnSpawnBsubDocker(trnBackEnd,dlTrnType,modelChain,...
%             'trnCompleteCbk',trainCompleteCbk2,...
%             args{:});
%         case DLBackEnd.AWS
%           obj2.trnSpawnAWS(trnBackEnd,dlTrnType,modelChain,'wbObj',wbObj);
%         otherwise
%           assert(false);
%       end
%       
%       % Nothing should occur here as failed trnSpawn* will early return
    
    
    function trainCompleteCbkStg1(obj,src,evt)
      obj.trnDoneStage1 = true;
      if obj.trnDoneStage2
        obj.trainStoppedCbk();
      end
    end
    
    function trainCompleteCbkStg2(obj,src,evt)
      obj.trnDoneStage2 = true;
      if obj.trnDoneStage1
        obj.trainStoppedCbk();
      end
    end
    
    function dmcs = trnGetDMCs(obj)
      dmcs = [obj.stage1Tracker.trnLastDMC(:); obj.trnLastDMC(:)];
    end
    
    function tc = getTrackerClassAugmented(obj2)
      obj1 = obj2.stage1Tracker;
      tc = {class(obj2) ...
        {'trnNetType' obj1.trnNetType 'trnNetMode' obj1.trnNetMode} ...
        {'trnNetType' obj2.trnNetType 'trnNetMode' obj2.trnNetMode} ...
        };
    end
    
    function s = getSaveToken(obj)
      s.stg1 = obj.stage1Tracker.getSaveToken();
      s.stg2 = getSaveToken@DeepTracker(obj);
    end
    
    function loadSaveToken(obj,s)
      s1 = DeepTracker.modernizeSaveToken(s.stg1);
      s2 = DeepTracker.modernizeSaveToken(s.stg2);
      % Important that this line occurs first, as DeepTracker/loadSaveToken
      % calls initHook(). If the stage1Tracker is loaded first, then it
      % gets cleared out.
      loadSaveToken@DeepTracker(obj,s2);
      loadSaveToken@DeepTracker(obj.stage1Tracker,s1);
    end
    
    function updateDLCache(obj,dlcachedir)
      updateDLCache@DeepTracker(obj,dlcachedir);
      updateDLCache@DeepTracker(obj.stage1Tracker,dlcachedir);
    end
    
  end
  
  methods (Static)
    
    function trkClsAug = getTrackerInfos
      % Currently-available TD trackers. Can consider moving to eg yaml later.
      trkClsAug = { ...
          {'DeepTrackerTopDown' ...
            {'trnNetType' DLNetType.multi_mdn_joint_torch ...
             'trnNetMode' DLNetMode.multiAnimalTDDetectHT} ...
            {'trnNetType' DLNetType.mdn_joint_fpn ...
             'trnNetMode' DLNetMode.multiAnimalTDPoseHT} ...
          }; ...
          {'DeepTrackerTopDown' ...
            {'trnNetType' DLNetType.detect_mmdetect ...
             'trnNetMode' DLNetMode.multiAnimalTDDetectObj} ...
            {'trnNetType' DLNetType.mdn_joint_fpn ...
             'trnNetMode' DLNetMode.multiAnimalTDPoseObj} ...
          }; ...
        };
    end
    
    function [codestr,containerName] = tdTrainCodeGenDockerDMC(tfSerial,...
        backend,dmcs,trnCmdType,mntPaths,gpuids)
      
      if tfSerial
        assert(isscalar(gpuids));
      else
        assert(numel(gpuids)==2);
      end
      
      assert(~any([dmcs.doSplit]));
            
      % where dmc1 is used here, it should matter whether dmcs(1) or dmcs(2) is used.
      dmc1 = dmcs(1);      
      baseargs0 = {...
        'maTopDown' true ... % missing: 'maTopDownStage'
        'maTopDownStage1NetType' dmcs(1).netType ...
        'maTopDownStage1NetMode' dmcs(1).netMode};
      args = { backend,...
        dmc1.modelChainID,dmc1.trainID,dmc1.lblStrippedLnx,...
        dmc1.rootDir,dmc1.errfileLnx,dmcs(2).netType,dmcs(2).netMode,...
        trnCmdType,dmc1.view+1,mntPaths }; 

      if tfSerial
        % for stage==0/serial, netType/Mode passed in regular arguments are
        % for stage 2.        
        stg = 0;
      else
        stg = [1;2];
      end
      assert(numel(gpuids)==numel(stg));
      
      [codestr,containerName] = arrayfun(...
        @(zstg,zgpuid) DeepTracker.trainCodeGenDocker(...
          args{:},zgpuid,'baseArgs',[baseargs0 {'maTopDownStage' zstg}]),...
        stg(:),gpuids(:),'uni',0);
    end
    
    function codestr = tdTrainCodeGenSSHBsubSingDMC(...
        tfSerial,aptroot,dmcs,trnCmdType,bsubargs,singargs)
      % ssh/bsub codegen for top-down
      %
      % Knows how to generate either a single codestr for serial train or 
      % two codestrs for parallel train
      %
      % tfSerial: true => codegen for serial train (single GPU). 
      %             codestr is [1] cellstr
      %           false => " parallel (2 GPUs). codestr is [2] cellstr
      %       
      % codestr: either [1] or [2] cellstr
      
      assert(~any([dmcs.doSplit]));
            
      % where dmc1 is used here, it should matter whether dmcs(1) or dmcs(2) is used.
      dmc1 = dmcs(1); 
      if isempty(aptroot)
        aptroot = dmc1.dirAptRootLnx;
      end

      repoSSscriptLnx = [aptroot '/matlab/repo_snapshot.sh'];
      repoSScmd = sprintf('"%s" "%s" > "%s"',repoSSscriptLnx,aptroot,...
        dmc1.aptRepoSnapshotLnx);
      prefix = [DLBackEndClass.jrcprefix '; ' repoSScmd];
      
      baseargs0 = {...
        'view' dmc1.view+1 'trainType' trnCmdType ...
        'deepnetroot' [aptroot '/deepnet'] ...
        'maTopDown' true ... % missing: 'maTopDownStage'
        'maTopDownStage1NetType' dmcs(1).netType ...
        'maTopDownStage1NetMode' dmcs(1).netMode};

      if tfSerial
        % for stage==0/serial, netType/Mode passed in regular arguments are
        % for stage 2.
        baseargs = [baseargs0 {'maTopDownStage' 0}];
        args = { ...
          dmc1.modelChainID,dmc1.lblStrippedLnx,...
          dmc1.rootDir,dmc1.errfileLnx,dmcs(2).netType,dmcs(2).netMode,...
          'singargs',singargs,'sshargs',{'prefix' prefix},...
          'bsubArgs',[bsubargs {'outfile' dmc1.trainLogLnx}]};
        codestr = DeepTracker.trainCodeGenSSHBsubSing(args{:},'baseArgs',baseargs);
        codestr = {codestr};
      else
        codestr = cell(2,1);
        for stg=1:2
          baseargs = [baseargs0 {'maTopDownStage' stg}];
          dmcS = dmcs(stg);
          args = { ...
            dmcS.modelChainID,dmcS.lblStrippedLnx,...
            dmcS.rootDir,dmcS.errfileLnx,dmcs(2).netType,dmcs(2).netMode,...
            'singargs',singargs,'sshargs',{'prefix' prefix},...
            'bsubArgs',[bsubargs {'outfile' dmcS.trainLogLnx}]};
          codestr{stg} = DeepTracker.trainCodeGenSSHBsubSing(args{:},'baseArgs',baseargs);
        end
      end
    end

  end
 
end