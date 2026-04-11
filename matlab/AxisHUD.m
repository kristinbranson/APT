classdef AxisHUD < handle
% Axis heads-up display
%
% Contains:
% * Series of text labels like Tgt: 3
% * handedness indicator

% TODO: Refactor for extensibility, just have structs/dicts instead of 
% hardcoding state/meths for tgt vs lblpoint vs susp etc.
  
  properties (Constant)
    txtXoff = 10
    txtHgt = 17
    txtWdh = 20  
    
    annoXoff = 0
    annoHgt = 34
    annoWdh = 50
    
    txtClrTarget = [1 0.6 0.784]
    txtClrLblPoint = [1 1 0]
    txtClrSusp = [1 1 1]
    txtClrTrklet = [1 1 1]
  end
  
  properties
    labelerController_
    labeler_
    axisHUDModel_
    hPanel  % scalar handle to the uipanel the HUD appears in
    hTxtTgt  % scalar handle to target text uicontrol
    hTxtLblPt  % scalar handle to some kind of text uicontrol
    hTxtSusp  % scalar handle to suspiciousness text uicontrol
    hTxtTrklet  % scalar handle to tracklet text uicontrol    
    hHandedAnno  % scalar annotation for handedness indicator
    hHandedListnr  % cell array of listeners to main axis .XDir and .YDir
  end
  
  properties (Dependent)
    hTxts  
      % col vec of handles to text uicontrols.  Contains all of hTxtTgt,
      % hTxtLblPt, hTxtSusp, hTxtTrklet, that are not empty.
    hasTgt  % scalar logical
    hasLblPt 
    hasSusp 
    hasTrklet
  end

  methods
    
    function obj = AxisHUD(labelerController, labeler, axisHUDModel, hPanel, hAxes)
      assert(ishandle(hPanel));
      obj.labelerController_ = labelerController ;
      obj.labeler_ = labeler ;
      obj.axisHUDModel_ = axisHUDModel ;
      obj.hPanel = hPanel;
      obj.initHandedAnno_();
      obj.clearHTxts_();
      
      lx = addlistener(hAxes,'XDir','PostSet',@(s,e)obj.cbkHandednessUpdate(s,e));
      ly = addlistener(hAxes,'YDir','PostSet',@(s,e)obj.cbkHandednessUpdate(s,e));
      obj.hHandedListnr = {lx ly};
      
      obj.cbkHandednessUpdate([],struct('AffectedObject',hAxes)); % initialize
    end

    function result = get.hasTgt(obj)
      result = obj.axisHUDModel_.hasTgt ;
    end  % function

    function result = get.hasLblPt(obj)
      result = obj.axisHUDModel_.hasLblPt ;
    end  % function

    function result = get.hasSusp(obj)
      result = obj.axisHUDModel_.hasSusp ;
    end  % function

    function result = get.hasTrklet(obj)
      result = obj.axisHUDModel_.hasTrklet ;
    end  % function
    
    function delete(obj)
      obj.clearHTxts_();

      deleteValidGraphicsHandles(obj.hHandedAnno);
      obj.hHandedAnno = [];
      
      for i=1:numel(obj.hHandedListnr)
        delete(obj.hHandedListnr{i});
      end
      obj.hHandedListnr = [];      
    end
        
    function initHandedAnno_(obj)
      parentpos = obj.hPanel.Position;
      y1 = parentpos(4) - obj.annoHgt; % just below top of hParent
            
      % Add a new textbox
      % hTxt: text (matlab.ui.control.UIControl) 
      % ytop (input): top/ceiling of axis. 
      % ytop (output): new top/ceiling after txtbox added.

      pos = [obj.annoXoff y1 obj.annoWdh obj.annoHgt];
      hAnn = annotation(obj.hPanel,'textbox',...
        'String','$\otimes z$',...
        'FontUnits','pixels',...
        'FontSize',26,...
        'FontWeight','bold',...
        'Units','pixels',...
        'Position',pos,...
        'Interpreter','latex',...
        'LineStyle','none',...
        'Color',[1 1 1],...
        'Tag','hud_handedness'...
        );
      obj.hHandedAnno = hAnn;
      %ytop = ytop - obj.txtHgt;      
    end
    
    function clearHTxts_(obj)
      delete(obj.hTxts);
      % obj.hTxts = matlab.ui.control.UIControl.empty(0,1);
      obj.hTxtTgt = [];
      obj.hTxtLblPt = [];
      obj.hTxtSusp = [];
      obj.hTxtTrklet = [];
    end
    
    function result = get.hTxts(obj)
      result = vertcat(obj.hTxtTgt, obj.hTxtLblPt, obj.hTxtSusp, obj.hTxtTrklet) ;
    end

    function updateReadoutFields(obj)
      % Update the readout fields.
      
      % if obj.hasTgt, tgtStr = obj.hTxtTgt.String; end
      % if obj.hasLblPt, lblPtStr = obj.hTxtLblPt.String; end
      % if obj.hasSusp, suspStr = obj.hTxtSusp.String; end
      % if obj.hasTrklet, trkletStr = obj.hTxtTrklet.String; end

      obj.clearHTxts_();
            
      %units0 = obj.hPanel.Units;
      %obj.hPanel.Units = 'pixels';
      parentpos = obj.hPanel.Position;
      %obj.hPanel.Units = units0;
      yOffset = parentpos(4) - obj.annoHgt; % just below anno
      if obj.hasTgt
        [obj.hTxtTgt,yOffset] = obj.createTextUIControl_(yOffset, obj.txtClrTarget, 'hud_tgt') ;
        obj.updateTarget_() ;
      end
      if obj.hasLblPt
        [obj.hTxtLblPt,yOffset] = obj.createTextUIControl_(yOffset, obj.txtClrLblPoint, 'hud_lblpt') ;
        obj.updateLblPoint_() ;
      end
      if obj.hasSusp
        obj.hTxtSusp = obj.createTextUIControl_(yOffset, obj.txtClrSusp, 'hud_susp') ;
        obj.updateSusp_() ;
      end
      if obj.hasTrklet
        obj.hTxtTrklet = obj.createTextUIControl_(yOffset, obj.txtClrTrklet, 'hud_trklet') ;
        obj.updateTrklet_() ;
      end
    end  % function
    
    function updateTarget_(obj)
      % Update the target readout.
      assert(obj.hasTgt) ;
      targetIndex = obj.labeler_.currTarget ;
      str = sprintf('tgt: %d', targetIndex) ;
      setStringAndFitWidthBang(obj.hTxtTgt, str) ;
    end

    function updateLblPoint_(obj)
      % Update the label-point readout.
      assert(obj.hasLblPt) ;
      nLblPts = obj.labeler_.lblCore.nPointSet ;
      iLblPt = obj.labeler_.lblCore.iSetWorking ;
      str = sprintf('Lbl pt: %d/%d', iLblPt, nLblPts) ;
      setStringAndFitWidthBang(obj.hTxtLblPt, str) ;
    end

    function updateSusp_(obj)
      % Update the suspiciousness readout.
      assert(obj.hasSusp) ;
      suspscore = obj.labeler_.currSusp ;
      str = sprintf('susp: %.10g', suspscore) ;
      setStringAndFitWidthBang(obj.hTxtSusp, str) ;
    end

    function updateTrklet_(obj)
      % Update the tracklet readout.
      assert(obj.hasTrklet) ;
      tracker = obj.labeler_.tracker ;
      if isempty(tracker) || isempty(tracker.trkVizer)
        str = '' ;
      else
        tvm = tracker.trkVizer ;
        if ~isa(tvm, 'TrackingVisualizerTrackletsModel')
          str = '' ;
        else
          iTrklet = tvm.currTrklet ;
          if ~(isscalar(iTrklet) && isfinite(iTrklet) && iTrklet>=1 && iTrklet<=numel(tvm.ptrx))
            str = '' ;            
          else
            trklet = tvm.ptrx(iTrklet).id ;
            ntrklettot = numel(tvm.ptrx) ;
            str = sprintf('trklet: %d (%d tot)', trklet, ntrklettot) ;
          end
        end
      end
      setStringAndFitWidthBang(obj.hTxtTrklet, str) ;
    end
 
    function [hTxt, ytop] = createTextUIControl_(obj, ytop, foreColor, tag)
      % Create a new textbox
      % hTxt: text (matlab.ui.control.UIControl)
      % ytop (input): top/ceiling of axis.
      % ytop (output): new top/ceiling after txtbox added.

      txtpos = [obj.txtXoff ytop-obj.txtHgt obj.txtWdh obj.txtHgt];
      hTxt = uicontrol(...
        'Style','text',...
        'HorizontalAlignment','left',...
        'Parent',obj.hPanel,...
        'FontUnits','pixels',...
        'FontName','Helvetica',...
        'FontSize',14,...
        'Units','pixels',...
        'Position',txtpos,...
        'ForegroundColor',foreColor,...
        'BackgroundColor',[0 0 0],...
        'Tag',tag);
      ytop = ytop - obj.txtHgt;
    end

    % function setHandednessViz(obj, tfviz)
    %   obj.hHandedAnno.Visible = onIff(tfviz) ;
    % end
    
    function setHandedness_(obj,trueForOut)
      if trueForOut
        obj.hHandedAnno.String = '$\odot z$';
      else
        obj.hHandedAnno.String = '$\otimes z$';        
      end
    end
    
    function cbkHandednessUpdate(obj,~,evt)
      ax = evt.AffectedObject;
      tfRightHanded = strcmp(ax.XDir,ax.YDir); % normal/normal or rev/rev
      obj.setHandedness_(tfRightHanded);
    end

    function layout(obj)
      % Position all HUD elements based on the current pixel size of hParent.

      parentPos = getpixelposition(obj.hPanel) ;
      parentH = parentPos(4) ;

      % Handedness annotation at top-left
      obj.hHandedAnno.Position = ...
        [obj.annoXoff, parentH - obj.annoHgt, obj.annoWdh, obj.annoHgt] ;

      % Text labels stack downward from just below the annotation
      y = parentH - obj.annoHgt ;
      hTxts = obj.hTxts ;
      for i = 1 : numel(hTxts)
        pos = hTxts(i).Position ;
        pos(1) = obj.txtXoff ;
        pos(2) = y - obj.txtHgt ;
        pos(4) = obj.txtHgt ;
        hTxts(i).Position = pos ;
        y = y - obj.txtHgt ;
      end
    end  % function
  end

end  % classdef
