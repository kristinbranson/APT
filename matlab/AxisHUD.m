classdef AxisHUD < handle
% Axis heads-up display
%
% Contains:
% * Series of text labels like Tgt: 3
% * handedness indicator

% TODO: Refactor for extensibility, just have structs/dicts instead of 
% hardcoding state/meths for tgt vs lblpoint vs susp etc.
  
  properties % basically Constant
    txtXoff = 10;
    txtHgt = 17;
    txtWdh = 130;  
    
    annoXoff = 0;
    annoHgt = 34;
    annoWdh = 50;
    
    txtClrTarget = [1 0.6 0.784];
    txtClrLblPoint = [1 1 0];
    txtClrSusp = [1 1 1];
    txtClrTrklet = [1 1 1];
    
    tgtFmt = 'tgt: %d';
    lblPointFmt = 'Lbl pt: %d/%d';
    suspFmt = 'susp: %.10g';
    trkletFmt = 'trklet: %d (%d tot)';
  end
  
  properties
    hParent; % scalar handle
    hTxts; % col vec of handles to text uicontrols
    hTxtTgt; % scalar handle (for convenience; owned by hTxts)
    hTxtLblPt; 
    hTxtSusp;
    hTxtTrklet;
    
    hHandedAnno; % scalar annotation for handedness indicator
    hHandedListnr; % cell array of listeners to main axis .XDir and .YDir
%     hAx; % axis for handedness
    
    hasTgt; % scalar logical
    hasLblPt; 
    hasSusp; 
    hasTrklet;
  end
  
  methods
    
    function obj = AxisHUD(h,hax)
      assert(ishandle(h));
      obj.hParent = h;
      obj.initHandedAnno();
      obj.initHTxts();
      obj.hasTgt = false;
      obj.hasLblPt = false;
      obj.hasSusp = false;
      obj.hasTrklet = false;
      
      lx = addlistener(hax,'XDir','PostSet',@(s,e)obj.cbkHandednessUpdate(s,e));
      ly = addlistener(hax,'YDir','PostSet',@(s,e)obj.cbkHandednessUpdate(s,e));
      obj.hHandedListnr = {lx ly};
      
      obj.cbkHandednessUpdate([],struct('AffectedObject',hax)); % initialize
%       obj.hAx = hax;
    end
    
    function delete(obj)
      obj.initHTxts();

      deleteValidGraphicsHandles(obj.hHandedAnno);
      obj.hHandedAnno = [];
      
      for i=1:numel(obj.hHandedListnr)
        delete(obj.hHandedListnr{i});
      end
      obj.hHandedListnr = [];      
    end
    
    function initHandedAnno(obj)
      units0 = obj.hParent.Units;
      obj.hParent.Units = 'pixels';
      parentpos = obj.hParent.Position;
      obj.hParent.Units = units0;
      y1 = parentpos(4) - obj.annoHgt; % just below top of hParent
            
      % Add a new textbox
      % hTxt: text (matlab.ui.control.UIControl) 
      % ytop (input): top/ceiling of axis. 
      % ytop (output): new top/ceiling after txtbox added.

      pos = [obj.annoXoff y1 obj.annoWdh obj.annoHgt];
      hAnn = annotation(obj.hParent,'textbox',...
        'String','$\otimes z$',...
        'FontUnits','pixels',...
        'FontSize',26,...
        'FontWeight','bold',...
        'Units','pixels',...
        'Position',pos,...
        'Interpreter','latex',...
        'LineStyle','none',...
        'Color',[1 1 1]...
        );
      hAnn.Units = 'normalized';
      hAnn.FontUnits = 'normalized';
      
      obj.hHandedAnno = hAnn;
      %ytop = ytop - obj.txtHgt;      
    end
    
    function initHTxts(obj)
      delete(obj.hTxts);
      obj.hTxts = matlab.ui.control.UIControl.empty(0,1);
      obj.hTxtTgt = [];
      obj.hTxtLblPt = [];
      obj.hTxtSusp = [];
      obj.hTxtTrklet = [];
    end
    
    function updateReadoutFields(obj,varargin)
      % Like setReadoutFields(), but preserves existing readout/strings
      % as applicable
      
      if obj.hasTgt, tgtStr = obj.hTxtTgt.String; end
      if obj.hasLblPt, lblPtStr = obj.hTxtLblPt.String; end
      if obj.hasSusp, suspStr = obj.hTxtSusp.String; end
      if obj.hasTrklet, trkletStr = obj.hTxtTrklet.String; end
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
      if obj.hasTrklet && exist('trkletStr','var')>0
        obj.hTxtTrklet.String = trkletStr;
      end
    end
    
    function setReadoutFields(obj,varargin)
      % obj.setReadoutFields('hasTgt',val,'hasLblPt',val,'hasSusp',val)
      %
      % Clears any existing HUD readouts/strings

      obj.initHTxts();
      
      [obj.hasTgt,obj.hasLblPt,obj.hasSusp,obj.hasTrklet] = ...
        myparse(varargin,...
        'hasTgt',obj.hasTgt,...
        'hasLblPt',obj.hasLblPt,...
        'hasSusp',obj.hasSusp,...
        'hasTrklet',obj.hasTrklet ...
        );

      units0 = obj.hParent.Units;
      obj.hParent.Units = 'pixels';
      parentpos = obj.hParent.Position;
      obj.hParent.Units = units0;
      y1 = parentpos(4) - obj.annoHgt; % just below anno
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
      if obj.hasTrklet
        obj.hTxtTrklet = obj.addTxt(y1,obj.txtClrTrklet);
        obj.hTxts(end+1,1) = obj.hTxtTrklet;
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
 
    function updateTrklet(obj,trklet,ntrklettot)
      assert(obj.hasTrklet);
      str = sprintf(obj.trkletFmt,trklet,ntrklettot);
      obj.hTxtTrklet.String = str;
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
    
    function setHandednessViz(obj,tfviz)
      obj.hHandedAnno.Visible = onIff(tfviz);
    end
    
    function setHandedness(obj,trueForOut)
      if trueForOut
        obj.hHandedAnno.String = '$\odot z$';
      else
        obj.hHandedAnno.String = '$\otimes z$';        
      end
    end
    
    function cbkHandednessUpdate(obj,src,evt)
      ax = evt.AffectedObject;
      tfRightHanded = strcmp(ax.XDir,ax.YDir); % normal/normal or rev/rev
      obj.setHandedness(tfRightHanded);
    end
  end
  
end

