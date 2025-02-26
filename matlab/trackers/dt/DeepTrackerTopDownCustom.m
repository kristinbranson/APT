classdef DeepTrackerTopDownCustom < DeepTrackerTopDown
  % Extending DeepTrackerTopDown to allow for Custom first stage and second
  % stage trackers
  
  properties
    valid
  end
  
  methods
    function v = getAlgorithmNameHook(obj)
      short_type_string = fif(strcmp(obj.topDownTypeStr, 'head/tail'), 'ht', 'bbox') ;
      v = sprintf('ma_top_down_custom_%s_%s_%s',...
                  short_type_string,...
                  obj.stage1Tracker.trnNetMode.shortCode,...
                  obj.trnNetMode.shortCode);
    end

    function v = getAlgorithmNamePrettyHook(obj)
      v = sprintf('Top Down (%s) Custom: %s + %s',...
                  obj.topDownTypeStr,...
                  obj.stage1Tracker.trnNetType.displayString,...
                  obj.trnNetType.displayString);
    end

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
      [stg1net,stg1mode] = ...
        myparse(stg1ctorargs_in, ...
                'trnNetType',[], ...
                'trnNetMode',DLNetMode.multiAnimalTDDetectHT);
      [stg2net,stg2mode] = ...
        myparse(stg2ctorargs_in, ...
                'trnNetType',[], ...
                'trnNetMode',DLNetMode.multiAnimalTDPoseHT);
      [~, valid] = ...
        myparse(varargin, ...
                'prev_tracker',[],...
                'valid', true);
      
      if lObj.silent || ~valid
        % Use default when silent -- for testing and initial dummy tracker
        def_td_info = DeepTrackerTopDown.getTrackerInfos;
        if stg1mode == DLNetMode.multiAnimalTDDetectHT
          stg1ctorargs = def_td_info{1}{2};
          stg2ctorargs = def_td_info{1}{3};
        else
          stg1ctorargs = def_td_info{2}{2};
          stg2ctorargs = def_td_info{2}{3};
        end
        
      else
        
        dlnets = enumeration('DLNetType');
        isma = [dlnets.isMultiAnimal];
        stg2nets = dlnets(~isma);
        
        is_bbox = false(1,numel(dlnets));
        for dndx = 1:numel(dlnets)
          
          if  dlnets(dndx).isMultiAnimal && ...
              startsWith(char(dlnets(dndx)),'detect_')
            is_bbox(dndx) = true;
          else
            is_bbox(dndx) = false;
          end
        end
        
        stg1nets_ht = dlnets(isma & ~is_bbox);
        stg1nets_bbox = dlnets(isma & is_bbox);
        if stg1mode == DLNetMode.multiAnimalTDDetectHT
          stg1nets = stg1nets_ht;
        else
          stg1nets = stg1nets_bbox;
        end
        if isempty(stg1net)
          [stg1net, stg2net] = DeepTrackerTopDownCustom.get_nets_ui(lObj,stg1nets,stg2nets);
        end

        if isempty(stg1net)
          def_td_info = DeepTrackerTopDown.getTrackerInfos;
          if stg1mode == DLNetMode.multiAnimalTDDetectHT
            stg1ctorargs = def_td_info{1}{2};
            stg2ctorargs = def_td_info{1}{3};
          else
            stg1ctorargs = def_td_info{2}{2};
            stg2ctorargs = def_td_info{2}{3};
          end
          valid = false;
        else
          stg1ctorargs = {'trnNetMode' stg1mode ...
            'trnNetType' stg1net};
          stg2ctorargs = {'trnNetMode' stg2mode ...
            'trnNetType' stg2net};        
        end
      end
      
      obj@DeepTrackerTopDown(lObj,stg1ctorargs,stg2ctorargs);
      obj.valid = valid;
      
    end  % function

    function ctorargs = get_constructor_args_to_match_stage_types(obj)  % constant method
      % Get the arguments that, when passed to the DeepTrackerTopDownCustom
      % constructor, will cause the constructor to return a newly-minted tracker
      % with the same stage types as prev_tracker.
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
    end
    
    function [stg1net, stg2net] = get_nets_ui(lObj, stg1nets, stg2nets)
        stg1nets_str = {};
        for ndx = 1:numel(stg1nets)
          stg1nets_str{ndx} = stg1nets(ndx).displayString;
        end
        stg2nets_str = {};
        for ndx = 1:numel(stg2nets)
          stg2nets_str{ndx} = stg2nets(ndx).displayString;
        end
        f = uifigure('Name','Networks for custom 2 Stage',...
          'Units','pixels','Position',[100,100,500,300]);
        centerOnParentFigure(f,lObj.hFig);
        s1_list = uilistbox(f,...
          'Position',[10 60 235 200],...
          'Items',stg1nets_str); %, ...
          %'ValueChangedFcn',@set_stg1net);
        
        s2_list = uilistbox(f,...
          'Position',[255 60 235 200],...
          'Items',stg2nets_str);
%         ,...
%           'ValueChangedFcn',@set_stg2net);
        uilabel(f,...
        'Text','Networks for stage 1 (Detection)',...
    'Position',[10 265 235 20]);
        uilabel(f,...
        'Text','Networks for stage 2 (Pose)',...
    'Position',[255 265 235 20]);
        pb_cancel = uibutton(f,'push',...
          'Position',[50 10 150 40],...
          'text','Cancel', ....
          'ButtonPushedFcn',@pb_cancel_callback);

        pb_apply = uibutton(f,'push',...
          'Position',[300 10 150 40],...
          'text','Apply', ....
          'ButtonPushedFcn',@pb_apply_callback);

        uiwait(f);
        
        function pb_cancel_callback(src,event)
          stg1net = [];
          stg2net = [];
          uiresume(f);
          close(f);
        end

        function pb_apply_callback(src,event)
          stg1net_ndx = find(strcmp(s1_list.Items,s1_list.Value));
          stg1net = stg1nets(stg1net_ndx);
          stg2net_ndx = find(strcmp(s2_list.Items,s2_list.Value));
          stg2net = stg2nets(stg2net_ndx);
          uiresume(f);
          close(f);
        end

    end
        
  end
    
end