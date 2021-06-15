classdef DeepTrackerTopDown < DeepTracker
  % Top-down/two-stage DeepTracker
  %
  % Inheriting here but composing another option. (Lazy to write fwding 
  % methods etc). The base/superclass deeptracker state is for the stage2
  % tracker, as in multianimal-with-trx tracking. DeepTrackerTopDown adds 
  % an additional DeepTracker object to represent detection/stage1.
  
  % factors
  % trnpack: just want to gen 1. AND, put it in a folder.
  % monitor: prob best to just have one. new vizer or repurpose
  % double-yaxis as in views.
  % but, ideally, could retrain each stage indep.
  % if multple GPUs, spawn both stages. else, run serially. trncomplete
  %   triggers stage2.
  % track. stage1 needs to finish first. trkcomplete triggers stage2.
  %   trackres. both potentially interesting.
  
  % could we somehow use two without mucking up DeepTracker too mcuh?
  % - OK trnpack. trnspawn, allow specification of pre-gened tp.
  % - monitor. support indep retrain of stages. i guess try having 2
  % monitors not horrible. 
  % - params. just force a copy not horrible.
  % - GPUs: assume two or JRC first, so no GPU shortage.
  % - track: trkcomplete trigger.
  
  properties
    stage1Tracker
    
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
        obj.trnNetType.displayString,obj.stage1Tracker.trnNetType.displayString);
    end
    function v = get.isHeadTail(obj)
      v = obj.trnNetMode.isHeadTail;
    end
    function v = get.topDownTypeStr(obj)
      if obj.isHeadTail
        v = 'head/tail';
      else
        v = 'object detect';
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
    
    function genStrippedLblTrnPack(obj,dlLblFileLcl)
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
    
    function retrain(obj2,varargin)
      [wbObj,dlTrnType] = myparse(varargin,...
        'wbObj',[],...
        'dlTrnType',DLTrainType.New ...
        );
      
      obj1 = obj2.stage1Tracker;
      objs = {obj1 obj2};
      for stg=1:2
        if objs{stg}.bgTrnIsRunning
          error('Stage %d training is already in progress.',stg);
        end
        if objs{stg}.bgTrkIsRunning
          error('Stage %d tracking is in progress.',stg);
        end
      end
            
      lblObj = obj2.lObj;
      projname = lblObj.projname;
      if isempty(projname)
        error('Please give your project a name. The project name will be used to identify your trained models on disk.');
      end
      
      trnBackEnd = lblObj.trackDLBackEnd;
      fprintf('Top-down multianimal tracking\n');
      fprintf('Your stage 1 (detection) deep net type is: %s\n',char(obj1.trnNetType));
      fprintf('Your stage 2 (pose tracking) deep net type is: %s\n',char(obj2.trnNetType));
      fprintf('Your training backend is: %s\n',char(trnBackEnd.type));
      %fprintf('Your training vizualizer is: %s\n',obj2.bgTrnMonitorVizClass);
      fprintf(1,'\n');
      
      for stg=1:2
        o = objs{stg};
        if o.isTrkFiles(),
          if isempty(o.skip_dlgs) || ~o.skip_dlgs
            qstr = sprintf('Stage %d: Tracking results exist for previous deep trackers. When training stops, these will be deleted. Continue training?',stg);
            res = questdlg(qstr,'Continue training?','Yes','No','Cancel','Yes');
            if ~strcmpi(res,'Yes'),
              return;
            end
          end
        end

        o.setAllParams(lblObj.trackGetParams());

        if isempty(o.sPrmAll)
          error('No tracking parameters have been set.');
        end

        o.bgTrnReset();
%         if ~isempty(oldVizObj),
%           delete(oldVizObj);
%         end
      end
      
      modelChain0 = obj2.trnName;
      dlTrnType = DLTrainType.New;
      switch dlTrnType
        case DLTrainType.New
          modelChain = datestr(now,'yyyymmddTHHMMSS');
          if ~isempty(modelChain0)
            assert(~strcmp(modelChain,modelChain0));
            fprintf('Training new model %s.\n',modelChain);
          end
%         case {DLTrainType.Restart DLTrainType.RestartAug}
%           if isempty(modelChain0)
%             error('Model has not been trained.');
%           end
%           modelChain = modelChain0;
%           fprintf('Restarting train on model %s.\n',modelChain);
        otherwise
          assert(false);
      end
            
      trainID = modelChain;
      dmcDummy = DeepModelChainOnDisk(...
        'rootDir',obj2.lObj.DLCacheDir,...
        'projID',obj2.lObj.projname,...
        'modelChainID',modelChain,...
        'trainID',trainID ...
        );
      dlLblFileLcl = dmcDummy.lblStrippedLnx;
      obj2.genStrippedLblTrnPack(dlLblFileLcl);
      switch trnBackEnd.type        
        case {DLBackEnd.Bsub DLBackEnd.Conda DLBackEnd.Docker}          
          
          obj2.trnDoneStage1 = false;
          obj2.trnDoneStage2 = false;
          
          trainStartCbk1 = @(s,e)0; % "do nothing"
          % trainStartCbk2 = []; % No need to supply, use default
          trainCompleteCbk1 = @(s,e)obj2.trainCompleteCbkStg1(s,e);
          trainCompleteCbk2 = @(s,e)obj2.trainCompleteCbkStg2(s,e);                    
          args = {'wbObj' wbObj 'existingTrnPackSLbl',dlLblFileLcl};
          obj1.trnSpawnBsubDocker(trnBackEnd,dlTrnType,modelChain,...
            'trnStartCbk',trainStartCbk1,...
            'trnCompleteCbk',trainCompleteCbk1,...
            args{:});
          obj2.trnSpawnBsubDocker(trnBackEnd,dlTrnType,modelChain,...
            'trnCompleteCbk',trainCompleteCbk2,...
            args{:});
        case DLBackEnd.AWS
          obj2.trnSpawnAWS(trnBackEnd,dlTrnType,modelChain,'wbObj',wbObj);
        otherwise
          assert(false);
      end
      
      % Nothing should occur here as failed trnSpawn* will early return
    end
    
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
            {'trnNetType' DLNetType.multi_mdn_joint_torch 'trnNetMode' DLNetMode.multiAnimalTDDetectHT} ...
            {'trnNetType' DLNetType.mdn_joint_fpn 'trnNetMode' DLNetMode.multiAnimalTDPoseHT} ...
          }; ...
          };
    end    
  end
 
end