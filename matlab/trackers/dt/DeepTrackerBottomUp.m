classdef DeepTrackerBottomUp < DeepTracker

  methods
    function v = getAlgorithmNameHook(obj)
      %v = sprintf('MA Bottom Up');
      v = obj.trnNetType.shortString;
    end
    function v = getAlgorithmNamePrettyHook(obj)      
      v = sprintf('Bottom Up: %s',obj.trnNetType.displayString);
    end
    function tc = getTrackerClassAugmented(obj)
      tc = {class(obj) 'trnNetType' obj.trnNetType 'trnNetMode' obj.trnNetMode};
    end
  end
      
  methods (Static)
    function trkClsAug = getTrackerInfos()
      % Currently-available BU trackers. Can consider moving to eg yaml later.
      trkClsAug = { ...
          {'DeepTrackerBottomUp' 'trnNetType' DLNetType.multi_mdn_joint_torch 'trnNetMode' DLNetMode.multiAnimalBU};
          {'DeepTrackerBottomUp' 'trnNetType' DLNetType.multi_openpose 'trnNetMode' DLNetMode.multiAnimalBU};
          {'DeepTrackerBottomUp' 'trnNetType' DLNetType.multi_cid 'trnNetMode' DLNetMode.multiAnimalBU};
          {'DeepTrackerBottomUp' 'trnNetType' DLNetType.multi_dekr 'trnNetMode' DLNetMode.multiAnimalBU};
          };
    end    
  end

end
