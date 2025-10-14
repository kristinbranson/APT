classdef TrackerCreateInfo
  % Represents the information needed to create a 'fresh' tracker.
  % This is meant to replace the "augmented class" cell array data structure
  % that used to be used for such info.  Having it be a proper class allowed
  % better type checking and leads to more readable code.  However, we still use
  % the cell-array-style structure when saving a LabelTracker, to avoid
  % saving instances of custom classes.

  properties (SetAccess=immutable)
    className  % old-school string
    netType  % 1 x stageCount
    netMode  % 1 x stageCount
    % moreConstructorArgs  % row cell array of old-school strings
  end

  methods
    function obj = TrackerCreateInfo(className, netTypes, netModes)
      % Constructor

      % Type-check args
      assert(isOldSchoolString(className)) ;
      assert(isa(netTypes, 'DLNetType') && isrow(netTypes)) ;
      assert(isa(netModes, 'DLNetMode') && isrow(netModes)) ;            
      netTypeCount = numel(netTypes) ;
      netModeCount = numel(netModes) ;
      assert(netTypeCount==netModeCount && (1<=netTypeCount) && (netTypeCount<=2)) ;
      
      % Assign things
      obj.className = className ;
      obj.netType = netTypes ;
      obj.netMode = netModes ;
    end

    function result = constructorArgs(obj)
      % Synthesize the full list of constructor arguments
      % if ~obj.valid()
      %   error('Can''t synthesize constructor args for an invalid TrackerCreateInfo') ;
      % end
      stageCount = numel(obj.netMode);
      if stageCount == 1
          result = {'trnNetType', obj.netType, 'trnNetMode', obj.netMode} ;
      else
        result = cell(1,0);
        for i = 1 : stageCount 
          result = horzcat(result, {{'trnNetType', obj.netType(i), 'trnNetMode', obj.netMode(i)}}) ; %#ok<AGROW>
        end
      end
    end  % function

    % function result = valid(obj)
    %   % DeepTrackerTopDownCustom TCIs have an empty .netTypes to start, and are thus
    %   % invalid.  This checks that the number ot .netTypes equals the number of
    %   % .netModes, and that they are either 1 or 2.
    %   netTypeCount = numel(obj.netType) ;
    %   netModeCount = numel(obj.netMode) ;
    %   result = (netTypeCount==netModeCount) && (1<=netTypeCount) && (netTypeCount<=2) ;
    % end  % function    
  end  % methods

  methods (Static)
    function result = fromNetTypes(netTypes, isMA)
      % A TrackerCreateInfo object specifies the exact subclass of LabelTracker to
      % be used to create a de novo tracker, along with how many stage it has, the
      % DLNetTypes of each stage, and the DLNetModes of each stage.  All these
      % things can be inferred the (one- or two-element) array of DLNetTypes, and
      % whether the current project is single- or multi-animal.  This function
      % performs that inference, returning a scalar TrackerCreateInfo that can be
      % handed to LabelTracker.create() to create a de novo tracker.
      assert(isShortDLNetTypesRowArray(netTypes));
      assert(islogical(isMA) && isscalar(isMA));
      if isMA
        if isscalar(netTypes)
          % Must be bottom-up          
          netType = netTypes;
          className = 'DeepTrackerBottomUp';
          netMode = DLNetMode.multiAnimalBU;
          result = TrackerCreateInfo(className, netType, netMode);
        else
          % Must be top-down
          % Most possible cases are handles by DeepTrackTopDownCustom, but a few are
          % handled by DeepTrackerTopDown
          className = 'DeepTrackerTopDown' ;
          % Determine whether bbox-style or head/tail-style object detection is being
          % used by looking at the first (detection) stage.
          if startsWith(char(netTypes(1)), 'detect')
            % If the stage 1 netType name starts with 'detect', it's one of the bbox-style
            % (aka object-style) detection networks, so stage 2 must be that style too.
            netModes = [ DLNetMode.multiAnimalTDDetectObj DLNetMode.multiAnimalTDPoseObj ];
          else
            % Stage 1 netType is not a bbox-style network, so must be head-tail.
            netModes = [ DLNetMode.multiAnimalTDDetectHT DLNetMode.multiAnimalTDPoseHT ];
          end
          result = TrackerCreateInfo(className, netTypes, netModes);
        end  % if iscalar(netTypes)
      else
        % Project is single-animal
        assert(isscalar(netTypes));
        netType = netTypes;
        if netType.isMultiAnimal
          error('For a functionally single-animal project, the netType cannot be multi-animal');
        end
        result = TrackerCreateInfo('DeepTracker', netType, DLNetMode.singleAnimal);
      end  % if isMA
      assert(isa(result, 'TrackerCreateInfo') && isscalar(result));
    end  % function

    function result = fromTCICellArray(ca, isMA)
      % Create a TrackerCreateInfo object from a cell array that represents one.
      className = ca{1} ;
      if isMA
        if strcmp(className, 'DeepTrackerBottomUp')
          netType = ca{3} ;
          result = TrackerCreateInfo('DeepTracker', netType, DLNetMode.multiAnimalBU) ;          
        elseif strcmp(className, 'DeepTrackerTopDown') || strcmp(className, 'DeepTrackerTopDownCustom')
          % The class DeepTrackerTopDownCustom doesn't exist anymore, now handled by
          % DeepTrackerTopDown.  But check for it to handle legacy projects.
          stage1NetTypeAndNetMode = ca{2} ;
          stage2NetTypeAndNetMode = ca{3} ;        
          stage1NetType = stage1NetTypeAndNetMode{2} ;
          stage1NetMode = stage1NetTypeAndNetMode{4} ;
          stage2NetType = stage2NetTypeAndNetMode{2} ;
          stage2NetMode = stage2NetTypeAndNetMode{4} ;
          netTypes = [ stage1NetType stage2NetType ] ;
          netModes = [ stage1NetMode stage2NetMode ] ;          
          result = TrackerCreateInfo('DeepTrackerTopDown', netTypes, netModes) ;
        else
          error('Unknown class name ''%s''', className) ;
        end
      else
        % Single-animal project
        netType = ca{3} ;
        result = TrackerCreateInfo('DeepTracker', netType, DLNetMode.singleAnimal) ;
      end
    end  % function
    
  end  % methods (Static)
end  % classdef
