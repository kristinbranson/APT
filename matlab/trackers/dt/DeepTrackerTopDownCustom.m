classdef DeepTrackerTopDownCustom < DeepTrackerTopDown
  % Extending DeepTrackerTopDown to allow for Custom first stage and second
  % stage trackers
  
  properties
    valid
  end
  
  methods
    function v = getAlgorithmNameHook(obj)
      v = sprintf('MA Top Down (Custom)');%,obj.trnNetMode.shortCode,...
      %        obj.stage1Tracker.trnNetMode.shortCode);
    end
    function v = getAlgorithmNamePrettyHook(obj)
      v = sprintf('Top-Down (%s) Custom',obj.topDownTypeStr);
    end
  end
  
  methods
    
    function obj = DeepTrackerTopDownCustom(lObj,...
        stg1ctorargs_in,stg2ctorargs_in, varargin)
      [stg1mode, stg1net] = myparse(stg1ctorargs_in, ...
        'trnNetMode',DLNetMode.multiAnimalTDDetectHT,...
        'stg1net',[]);
      [stg2mode, stg2net] = myparse(stg2ctorargs_in, ...
        'trnNetMode',DLNetMode.multiAnimalTDPoseHT,...
        'stg2net',[]);
      [prev_tracker, valid] = myparse(varargin,'prev_tracker',[],...
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
        [stg1net, stg2net] = DeepTrackerTopDownCustom.get_nets_ui(lObj,stg1nets,stg2nets);

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
      
    end
    
  end
  
  methods (Static)
    
    function trkClsAug = getTrackerInfos
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
    
    function use = use_prev(prev_tracker)
      use = false;
      if ~isempty(prev_tracker) && prev_tracker.valid
        prev_net_str = sprintf('%s (stage 1) and %s (stage 2)',...
          prev_tracker.stage1Tracker.trnNetType.displayString, ...
          prev_tracker.trnNetType.displayString );

        qstr = sprintf('Continue with previous custom tracker with %s?. Note that if you change, you will lose any previously trained tracker of this type.',prev_net_str );
        res = questdlg(qstr,....
          'Change custom 2 stage tracker',...
          'Continue','Change','Continue');
        if strcmp(res,'Continue')
          use = true;
          return
        end

      end
    end
    
    function ctorargs = get_args(prev_tracker)
      stg1type = prev_tracker.stage1Tracker.trnNetMode;
      stg2type = prev_tracker.trnNetMode;
      ctorargs = { {'trnNetMode' stg1type} 
                   {'trnNetMode' stg2type} 
                   };
    end
    
  end
    
end