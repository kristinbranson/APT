classdef AxisHUD < handle
  % Heads-up display for an axis; right now specifically for Labeler
  
  properties
    txtXoff = 10;
    txtHgt = 17;
    txtWdh = 130;  
    
    txtClrTarget = [1 0.6 0.784];
    txtClrLblPoint = [1 1 0];
    txtClrSusp = [1 1 1];
    
    tgtFmt = 'tgtID: %d';
    lblPointFmt = 'Lbl pt: %d/%d';
    suspFmt = 'susp: %.10g';
  end
  
  properties
    ax; % scalar axis handle    
    hTxts; % col vec of handles to text uicontrols
    hTxtTgt; % scalar handle (for convenience; owned by hTxts)
    hTxtLblPoint; 
    hTxtSusp;
    
    hasTgt; % scalar logical
    hasLblPt; 
    hasSusp; 
  end
  
  methods
    
    function obj = AxisHUD(ax)
      assert(isa(ax,'matlab.graphics.axis.Axes'));
      obj.ax = ax;
      obj.initTxts;
    end
    
    function delete(obj)
      obj.initTxts();
    end
    
    function initTxts(obj)
      delete(obj.hTxts);
      obj.hTxts = matlab.ui.control.UIControl.empty(0,1);
      obj.hTxtTgt = [];
      obj.hTxtLblPoint = [];
      obj.hTxtSusp = [];
    end
    
    function setReadoutFields(obj,varargin)
      % obj.setReadoutFields('hasTgt',val,'hasLblPt',val,'hasSusp',val)
      
      [obj.hasTgt,obj.hasLblPt,obj.hasSusp] = ...
        myparse(varargin,...
        'hasTgt',obj.hasTgt,...
        'hasLblPt',obj.hasLblPt,...
        'hasSusp',obj.hasSusp);
      
      obj.initTxts();
      units0 = obj.ax.Units;
      obj.ax.Units = 'pixels';
      axpos = obj.ax.Position;
      obj.ax.Units = units0;
      y1 = axpos(2)+axpos(4); % current 'top' of axis
      if obj.hasTgt
        [obj.hTxtTgt,y1] = obj.addTxt(y1,obj.txtClrTarget);
        obj.hTxts(end+1,1) = obj.hTxtTgt;
      end
      if obj.hasLblPt
        [obj.hTxtLblPoint,y1] = obj.addTxt(y1,obj.txtClrLblPoint);
        obj.hTxts(end+1,1) = obj.hTxtLblPoint;
      end
      if obj.hasSusp
        obj.hTxtSusp = obj.addTxt(y1,obj.txtClrSusp);
        obj.hTxts(end+1,1) = obj.hTxtSusp;
      end
    end
    
    function updateTarget(obj,tgtID)
      assert(obj.hasTgt)
      str = sprintf(obj.tgtFmt,tgtID);
      set(obj.hTxtTgt,'String',str);
    end
    
    function updateLblPoint(obj,iLblPt,nLblPts)
      assert(obj.hasLblPt);
      str = sprintf(obj.lblPointFmt,iLblPt,nLblPts);
      obj.hTxtLblPoint.String = str;      
    end
    
    function updateSusp(obj,suspscore)      
      assert(obj.hasSusp);
      str = sprintf(obj.suspFmt,suspscore);
      obj.hTxtSusp.String = str;
    end
    
    function [hTxt,ytop] = addTxt(obj,ytop,foreColor)
      % Add a new textbox
      % hTxt: text (matlab.ui.control.UIControl) 
      % ytop (input): top/ceiling of axis. 
      % ytop (output): new top/ceiling after txtbox added.

      txtpos = [obj.txtXoff ytop-obj.txtHgt obj.txtWdh obj.txtHgt];
      hTxt = uicontrol(...
        'Style','text',...
        'Parent',obj.ax.Parent,...
        'Units','pixels',...
        'Position',txtpos,...
        'ForegroundColor',foreColor,...
        'BackgroundColor',[0 0 0]);
      hTxt.Units = 'normalized';
      ytop = ytop - obj.txtHgt;
    end
    
  end
  
end

