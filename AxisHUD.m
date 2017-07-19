classdef AxisHUD < handle
% Axis heads-up display
  
  properties % basically Constant
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
    hParent; % scalar handle    
    hTxts; % col vec of handles to text uicontrols
    hTxtTgt; % scalar handle (for convenience; owned by hTxts)
    hTxtLblPt; 
    hTxtSusp;
    
    hasTgt; % scalar logical
    hasLblPt; 
    hasSusp; 
  end
  
  methods
    
    function obj = AxisHUD(h)
      assert(ishandle(h));
      obj.hParent = h;
      obj.initHTxts();
      obj.hasTgt = false;
      obj.hasLblPt = false;
      obj.hasSusp = false;
    end
    
    function delete(obj)
      obj.initHTxts();
    end
    
    function initHTxts(obj)
      delete(obj.hTxts);
      obj.hTxts = matlab.ui.control.UIControl.empty(0,1);
      obj.hTxtTgt = [];
      obj.hTxtLblPt = [];
      obj.hTxtSusp = [];
    end
    
    function updateReadoutFields(obj,varargin)
      % Like setReadoutFields(), but preserves existing readout/strings
      % as applicable
      
      if obj.hasTgt, tgtStr = obj.hTxtTgt.String; end
      if obj.hasLblPt, lblPtStr = obj.hTxtLblPt.String; end
      if obj.hasSusp, suspStr = obj.hTxtSusp.String; end
      obj.setReadoutFields(varargin{:});
      if obj.hasTgt && exist('tgtStr','var')>0
        obj.hTxtTgt.String = tgtStr;
      end
      if obj.hasLblPt && exist('lblPtStr','var')>0
        obj.hTxtLblPt.String = lblPtStr;
      end
      if obj.hasSusp && exist('suspStr','var')>0
        obj.hTxtSusp.String = suspStr;
      end
    end
    
    function setReadoutFields(obj,varargin)
      % obj.setReadoutFields('hasTgt',val,'hasLblPt',val,'hasSusp',val)
      %
      % Clears any existing HUD readouts/strings

      obj.initHTxts();
      
      [obj.hasTgt,obj.hasLblPt,obj.hasSusp] = ...
        myparse(varargin,...
        'hasTgt',obj.hasTgt,...
        'hasLblPt',obj.hasLblPt,...
        'hasSusp',obj.hasSusp);

      units0 = obj.hParent.Units;
      obj.hParent.Units = 'pixels';
      parentpos = obj.hParent.Position;
      obj.hParent.Units = units0;
      y1 = parentpos(4) - obj.txtHgt; % just below top of hParent
      if obj.hasTgt
        [obj.hTxtTgt,y1] = obj.addTxt(y1,obj.txtClrTarget);
        obj.hTxts(end+1,1) = obj.hTxtTgt;
      end
      if obj.hasLblPt
        [obj.hTxtLblPt,y1] = obj.addTxt(y1,obj.txtClrLblPoint);
        obj.hTxts(end+1,1) = obj.hTxtLblPt;
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
      obj.hTxtLblPt.String = str;      
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
        'HorizontalAlignment','left',...
        'Parent',obj.hParent,...
        'FontUnits','pixels',...
        'FontName','Helvetica',...
        'FontSize',14,...
        'Units','pixels',...
        'Position',txtpos,...
        'ForegroundColor',foreColor,...
        'BackgroundColor',[0 0 0]);
      hTxt.Units = 'normalized';
      hTxt.FontUnits = 'normalized';
      ytop = ytop - obj.txtHgt;
    end
    
  end
  
end

