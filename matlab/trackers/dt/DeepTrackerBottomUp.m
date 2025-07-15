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
    function [tf,loc] = isMemberTrnTypes(trntypes)
      % [tf,loc] = isMemberTrnTypes(trntypes)
      % Based on getTrackerInfos(), figure out if trntypes is a possible
      % instantiation for this class
      tf = false;
      loc = 0;
      if numel(trntypes) ~= 1,
        return;
      end
      infos = DeepTrackerTopDown.getTrackerInfos();
      for i = 1:numel(infos),
        if strcmp(infos{i}{3}.shortString,trntypes.shortString),
          tf = true;
          loc = i;
          return;
        end
      end
    end  % function
  end

end
