classdef AxesHighlightManager < handle
  % Handle highlighting of axes for 'special' frames, used by eg GT mode
  
  properties
    hAxs % vector of axes to which hiliting is applied
    tfHilite % scalar logical, current hilite state
    tfHilitePnl % scalar logical. If true, hilite panel; otherwise hilite axes
    
    sRegProps % (nested) struct of regular prop vals. mirrors HILITEPROPS
              % but with default/original vals
  end
  
  properties (Constant)
    ORANGE = [1 .6 .2];
  end
  properties
    HILITEPROPS = struct(...
      'axes',struct(...
        'XColor',AxesHighlightManager.ORANGE,...
        'YColor',AxesHighlightManager.ORANGE,...
        'LineWidth',3),...
      'uipanel',struct(...
        'ShadowColor',AxesHighlightManager.ORANGE,...
        'HighlightColor',AxesHighlightManager.ORANGE,...
        'BorderWidth',3)...
        );
  end
  
  methods
    
    function obj = AxesHighlightManager(hAxes)
      assert(isa(hAxes,'matlab.graphics.axis.Axes'));
      assert(~isempty(hAxes));
      
      % NOTE: Trx vs noTrx, Axes vs Panels
      % Atm, hasTrx-ness is not encoded at the project level. (It probably
      % should be). Ie, a project may have some movies with trx and some
      % without. So, the highlight manager must know how to handle both
      % with-trx hilighting and without-trx highlighting.
      
      obj.hAxs = hAxes;
      
      sAx = struct();
      flds = fieldnames(obj.HILITEPROPS.axes);
      for f=flds(:)',f=f{1}; %#ok<FXSET>
        sAx.(f) = hAxes(1).(f); 
      end
      
      sPnl = struct();
      flds = fieldnames(obj.HILITEPROPS.uipanel);
      for f=flds(:)',f=f{1}; %#ok<FXSET>
        sPnl.(f) = hAxes(1).Parent.(f); 
      end
      
      obj.sRegProps = struct('axes',sAx,'uipanel',sPnl);
      obj.tfHilitePnl = false;
      obj.tfHilite = false;
    end
    
    function setHighlightPanel(obj,tf)
      obj.setHighlight(false);
      obj.tfHilitePnl = tf;
    end
    
    function setHighlight(obj,tf)
      if tf~=obj.tfHilite
        if obj.tfHilitePnl
          assert(isscalar(obj.hAxs));
          h = obj.hAxs.Parent;
          propsFld = 'uipanel';
        else
          h = obj.hAxs;
          propsFld = 'axes';
        end
        
        if tf
          set(h,obj.HILITEPROPS.(propsFld));
        else
          set(h,obj.sRegProps.(propsFld));
        end
        obj.tfHilite = tf;
      end
    end
          
  end
  
end