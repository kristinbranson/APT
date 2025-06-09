classdef DeepTrackerTopDownCustom < DeepTrackerTopDown
  % Extending DeepTrackerTopDown to allow for Custom first stage and second
  % stage trackers
  
  properties
    valid  % true iff has both stages specified.  If false, this object acts as a sort of dummy, or placeholder.
  end
  
  methods
    function v = getAlgorithmNameHook(obj)
      short_type_string = fif(strcmp(obj.topDownTypeStr, 'head/tail'), 'ht', 'bbox') ;
      if obj.valid
        v = sprintf('ma_top_down_custom_%s_%s_%s',...
                    short_type_string,...
                    obj.stage1Tracker.trnNetMode.shortCode,...
                    obj.trnNetMode.shortCode);
      else
        v = sprintf('ma_top_down_custom_%s',...
                    short_type_string) ;
      end        
    end

    function v = getAlgorithmNamePrettyHook(obj)
      if obj.valid
        v = sprintf('Top Down (%s) Custom: %s + %s',...
                    obj.topDownTypeStr,...
                    obj.stage1Tracker.trnNetType.displayString,...
                    obj.trnNetType.displayString) ;
      else
        v = sprintf('Top Down (%s) Custom',...
                    obj.topDownTypeStr) ;
      end        
    end  % function

%     function v = getAlgorithmNameHook(obj)
%       v = sprintf('MA Top Down (Custom,%s,%s)', ...
%                   obj.trnNetMode.shortCode,...
%                   obj.stage1Tracker.trnNetMode.shortCode);
%     end
%
%     function v = getAlgorithmNamePrettyHook(obj)
%       v = sprintf('Top-Down (%s) Custom',obj.topDownTypeStr);
%     end
  end
  
  methods
    function obj = DeepTrackerTopDownCustom(lObj, stg1ctorargs_in, stg2ctorargs_in, varargin)
      % Is this going to be a fully-specified custom tracker (valid==true), or a
      % dummy one (valid==false)?
      valid = ...
        myparse(varargin, ...
                'valid', true) ;  
      if valid
        % For a valid custom tracker, things are simpler.
        stg1ctorargs = stg1ctorargs_in ;
        stg2ctorargs = stg2ctorargs_in ;
      else
        % Use defaults when this is a dummy tracker.
        def_td_info = DeepTrackerTopDown.getTrackerInfos() ;       
        [~,stg1mode] = ...
          myparse(stg1ctorargs_in, ...
                  'trnNetType',[], ...
                  'trnNetMode',DLNetMode.multiAnimalTDDetectHT);  % a bit awkward...
        if stg1mode == DLNetMode.multiAnimalTDDetectHT
          % Object detection
          stg1ctorargs = def_td_info{1}{2};
          stg2ctorargs = def_td_info{1}{3};
        else
          stg1ctorargs = def_td_info{2}{2};
          stg2ctorargs = def_td_info{2}{3};
        end        
      end

      % Finally, call the superclass constructor and store validity.
      obj@DeepTrackerTopDown(lObj,stg1ctorargs,stg2ctorargs);
      obj.valid = valid;      
    end  % function

    function ctorargs = get_constructor_args_to_match_stage_types(obj)  % constant method
      % Get the arguments that, when passed to the DeepTrackerTopDownCustom
      % constructor, will cause the constructor to return a newly-minted tracker
      % with the same stage types as obj.
      stg1type = obj.stage1Tracker.trnNetMode;
      stg2type = obj.trnNetMode;
      ctorargs = { {'trnNetMode' stg1type} 
                   {'trnNetMode' stg2type} 
                   };
    end  % function
    
  end  % methods
  
  methods (Static)
    
    function trkClsAug = getTrackerInfos()
      % Currently-available TD trackers. Can consider moving to eg yaml later.
      trkClsAug = { ...
          {'DeepTrackerTopDownCustom' ...
            {'trnNetMode' DLNetMode.multiAnimalTDDetectHT} ...
            {'trnNetMode' DLNetMode.multiAnimalTDPoseHT} ...
            'valid' false ...
          }; ...
          {'DeepTrackerTopDownCustom' ...
            {'trnNetMode' DLNetMode.multiAnimalTDDetectObj} ...
            {'trnNetMode' DLNetMode.multiAnimalTDPoseObj} ...
            'valid' false ...
          }; ...
        };
    end  % function   
        
  end  % methods (Static)
    
end  % classdef
