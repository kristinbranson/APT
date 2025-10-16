classdef DeepTrackerBottomUp < DeepTracker  
  % Represents a bottom-up tracker.  Bottom-up trackers have a single stage.

  methods
    function v = getAlgorithmNameHook(obj)
      %v = sprintf('MA Bottom Up');
      v = obj.trnNetType.shortString;
    end

    function v = getAlgorithmNamePrettyHook(obj)      
      v = sprintf('Bottom Up: %s',obj.trnNetType.displayString);
    end

    function result = trackerCreateInfo(obj)
      result = TrackerCreateInfo('DeepTrackerBottomUp', obj.trnNetType, obj.trnNetMode) ;
    end   
  end  % methods
      
  methods (Static)
    % function tcis = getTrackerInfos()
    %   % Currently-available BU trackers. Can consider moving to eg yaml later.
    %   tci1 = TrackerCreateInfo('DeepTrackerBottomUp', DLNetType.multi_mdn_joint_torch, DLNetMode.multiAnimalBU) ;
    %   tci2 = TrackerCreateInfo('DeepTrackerBottomUp', DLNetType.multi_openpose, DLNetMode.multiAnimalBU) ;
    %   tci3 = TrackerCreateInfo('DeepTrackerBottomUp', DLNetType.multi_cid, DLNetMode.multiAnimalBU) ;
    %   tci4 = TrackerCreateInfo('DeepTrackerBottomUp', DLNetType.multi_dekr, DLNetMode.multiAnimalBU) ;
    %   tcis = [ tci1 tci2 tci3 tci4 ] ;
    % end    

    % function [tf,loc] = isMemberTrnTypes(queryNetTypes)
    %   % [tf,loc] = isMemberTrnTypes(queryNetTypes)
    %   % Based on getTrackerInfos(), figure out if the net types queryNetTypes
    %   % are possible for an instance of this class.  On return, if tf is true then
    %   % loc is the index into getTrackerInfos() of the matching entry.
    % 
    %   assert(isShortDLNetTypesRowArray(queryNetTypes)) ;
    % 
    %   tf = false;
    %   loc = 0;
    %   if numel(queryNetTypes) ~= 1,
    %     return
    %   end
    %   queryNetType = queryNetTypes ;
    %   tcis = DeepTrackerTopDown.getTrackerInfos();
    %   for i = 1:numel(tcis),
    %     tci = tcis(i) ;
    %     if strcmp(tci.netType.shortString, queryNetType.shortString) ,
    %       tf = true;
    %       loc = i;
    %       return
    %     end
    %   end
    % end  % function
  end  % methods (Static)
end  % classdef
