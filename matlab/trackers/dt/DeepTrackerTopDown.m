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
    %forceSerial = false;
    
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
      short_type_string = fif(strcmp(obj.topDownTypeStr, 'head/tail'), 'ht', 'bbox') ;
      v = sprintf('ma_top_down_%s_%s_%s',...
                  short_type_string,...
                  obj.stage1Tracker.trnNetMode.shortCode,...
                  obj.trnNetMode.shortCode);
    end  % function

    function v = getAlgorithmNamePrettyHook(obj)
      v = sprintf('Top Down (%s): %s + %s',...
                  obj.topDownTypeStr,...
                  obj.stage1Tracker.trnNetType.displayString,...
                  obj.trnNetType.displayString);
    end  % function

    function v = getNetsUsed(obj)
      v = cellstr([obj.stage1Tracker.trnNetType; obj.trnNetType]);
    end  % function

    function v = getNumStages(obj)  %#ok<MANU>
      v = 2;
    end  % function

    function v = get.isHeadTail(obj)
      v = obj.trnNetMode.isHeadTail;
    end  % function

    function v = get.topDownTypeStr(obj)
      if obj.isHeadTail
        v = 'head/tail';
      else
        v = 'bbox';
      end
    end  % function
    
  end  % methods
    
  methods 

    function obj = DeepTrackerTopDown(lObj,stg1ctorargs,stg2ctorargs)
      obj@DeepTracker(lObj,stg2ctorargs{:});
      dt = DeepTracker(lObj,stg1ctorargs{:});
      obj.stage1Tracker = dt;      
    end
    
    function initHook(obj)
      initHook@DeepTracker(obj);
      obj.stage1Tracker.init();      
    end
    
    function setAllParams(obj,sPrmAll)
      obj.stage1Tracker.setAllParams(sPrmAll);
      setAllParams@DeepTracker(obj,sPrmAll);      
    end
    
    function tf = isTrkFiles(obj)
      tf = isTrkFiles@DeepTracker(obj) || obj.stage1Tracker.isTrkFiles();
    end

    function netType = getNetType(obj)
      netType = [obj.stage1Tracker.trnNetType,getNetType@DeepTracker(obj)];
    end
    function netMode = getNetMode(obj)
      netMode = [obj.stage1Tracker.trnNetMode,getNetMode@DeepTracker(obj)];
    end
    function iterFinal = getIterFinal(obj)
      sPrmGDStg1 = obj.sPrmAll.ROOT.MultiAnimal.Detect.DeepTrack.GradientDescent;
      iterFinal = [sPrmGDStg1.dl_steps,getIterFinal@DeepTracker(obj)];
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
    function s = getTrackSaveToken(obj)
      s = obj.getSaveToken();
      s.stg1.sPrmAll = APTParameters.all2TrackParams(s.stg1.sPrmAll,false);
      s.stg2.sPrmAll = APTParameters.all2TrackParams(s.stg2.sPrmAll,false);
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
    
    function setTrackParams(obj,sPrmTrack)
      setTrackParams@DeepTracker(obj,sPrmTrack);
      setTrackParams@DeepTracker(obj.stage1Tracker,sPrmTrack);
    end  % function

  end  % mehtods
  
  methods (Static)
    
    function tcis = getTrackerInfos()
      % Currently-available TD trackers. Can consider moving to eg yaml later.
      % trkClsAug = { ...
      %     {'DeepTrackerTopDown' ...
      %       {'trnNetType' DLNetType.multi_mdn_joint_torch ...
      %        'trnNetMode' DLNetMode.multiAnimalTDDetectHT} ...
      %       {'trnNetType' DLNetType.mdn_joint_fpn ...
      %        'trnNetMode' DLNetMode.multiAnimalTDPoseHT} ...
      %     }; ...
      %     {'DeepTrackerTopDown' ...
      %       {'trnNetType' DLNetType.detect_mmdetect ...
      %        'trnNetMode' DLNetMode.multiAnimalTDDetectObj} ...
      %       {'trnNetType' DLNetType.mdn_joint_fpn ...
      %        'trnNetMode' DLNetMode.multiAnimalTDPoseObj} ...
      %     }; ...
      %   };
      tci1 = TrackerCreateInfo('DeepTrackerBottomUp', ...
                               [ DLNetType.multi_mdn_joint_torch DLNetType.mdn_joint_fpn ], ...
                               [ DLNetMode.multiAnimalTDDetectHT DLNetMode.multiAnimalTDPoseHT ]) ;
      tci2 = TrackerCreateInfo('DeepTrackerBottomUp', ...
                               [ DLNetType.detect_mmdetect DLNetType.mdn_joint_fpn ], ...
                               [ DLNetMode.multiAnimalTDDetectObj DLNetMode.multiAnimalTDPoseObj ]) ;
      tcis = [ tci1 tci2 ] ;
    end  % function    

    function [tf,loc] = isMemberTrnTypes(queryNetTypes)
      % [tf,loc] = isMemberTrnTypes(queryNetTypes)
      % Based on getTrackerInfos(), figure out if queryNetTypes is a possible
      % instantiation for this class
      tf = false;
      loc = 0;
      if numel(queryNetTypes) ~= 2,
        return;
      end
      possibleTCIs = DeepTrackerTopDown.getTrackerInfos();
      for i = 1:numel(possibleTCIs),
        possibleTCI = possibleTCIs(i) ;
        if all(possibleTCI.netType == queryNetTypes)
          tf = true;
          loc = i;
          return
        end
      end
    end  % function

  end  % methods (Static)
 
end  % classdef
