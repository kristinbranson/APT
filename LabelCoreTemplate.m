classdef LabelCoreTemplate < LabelCore
  
  % DESCRIPTION
  % In Template mode, there is a set of template/"white" points on the
  % image at all times. (When starting, these points need to be either
  % specified or loaded/imported.) To label a frame, adjust the points as 
  % necessary and accept. Adjusted points are shown in colors (rather than
  % white).
  %
  % Points may also be Selected using hotkeys (0..9). When a point is
  % selected, the arrow-keys adjust the point as if by mouse. Mouse-clicks
  % on the image also jump the point immediately to that location.
  % Adjustment of a point in this way is identical in concept to
  % click-dragging.
  %
  %
  % IMPL NOTES
  % 2 basic states, Adjusting and Accepted
  % - Adjusting, unlabeled frame underneath
  % -- 'untouched' template points are white
  % -- 'touched' template points are colored
  % -- Accept writes to labeledpos
  % - Adjusting, labeled frame underneath
  % -- all pts colored/'touched'
  % -- labeledpos exists underneath and is not overwritten until Acceptance
  % - Accepted
  % -- all points colored.
  % -- Can go back into adjustment mode, but all points turn white as if from template.
  %
  % TRANSITIONS W/OUT TRX
  % Note: Once a template is created/loaded, there is a set of points (either
  %  all white/template, or partially template, or all colored) on the image
  %  at all times.
  % - New unlabeled frame
  % -- Label point positions are unchanged, but all points become white.
  %    If scrolling forward, the template from frame-1 will be used etc.
  % -- (Enhancement? not implemented) Accepted points from 'nearest'
  %    neighboring labeled frame (eg frm-1, or <frmLastVisited>, become
  %    white points in new frame
  % - New labeled frame
  % -- Previously accepted labels shown as colored points.
  %
  % TRANSITIONS W/TRX
  % - New unlabeled frame (current target)
  % -- Existing points are auto-aligned onto new frame. The existing
  %    points could represent previously accepted labels, or just the
  %    previous template state.
  % - New labeled frame (current target)
  % -- Previously accepted labels shown as colored points.
  % - New target (same frame), unlabeled
  % -- Last labeled frame for new target acts as template; points auto-aligned.
  % -- If no previously labeled frame for this target, current white points are aligned onto new target.
  % - New target (same frame), labeled
  % -- Previously accepted labels shown as colored points.
  
  properties
    iPtMove;     % scalar. Either nan, or index of pt being moved
    tfMoved;     % scalar logical; if true, pt being moved was actually moved
    tfAdjusted;  % nPts x 1 logical vec. If true, pt has been adjusted from template
    tfPtSel;   % nPts x 1 logical
    
    templatePointColor = [1 1 1];  % 1 x 3 RGB
    selectedPointMarker = 'x';
  end  
  
  methods
    
    function obj = LabelCoreTemplate(varargin)
      obj = obj@LabelCore(varargin{:});
    end
    
    function initHook(obj)
      obj.setRandomTemplate();
      obj.tfAdjusted = false(obj.nPts,1);
      obj.tfPtSel = false(obj.nPts,1);
      
      ppi = obj.ptsPlotInfo;
      obj.templatePointColor = ppi.TemplateMode.TemplatePointColor;
      obj.selectedPointMarker = ppi.TemplateMode.SelectedPointMarker;
    end
    
  end
  
  methods
    
    function newFrame(obj,iFrm0,iFrm1,iTgt)
      [tflabeled,lpos] = obj.labeler.labelPosIsLabeled(iFrm1,iTgt);
      if tflabeled
        obj.assignLabelCoords(lpos);
        obj.enterAccepted(false);
      else
        if obj.labeler.hasTrx
          % existing points are aligned onto new frame based on trx at
          % (currTarget,prevFrame) and (currTarget,currFrame)
          
          xy0 = obj.getLabelCoords();
          xy = LabelCore.transformPtsTrx(xy0,obj.labeler.trx(iTgt),iFrm0,obj.labeler.trx(iTgt),iFrm1);          
          obj.assignLabelCoords(xy,'tfClip',true);
        else
          % none, leave pts as-is
        end          
        obj.enterAdjust(true,false);
      end
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm)
      [tflabeled,lpos] = obj.labeler.labelPosIsLabeled(iFrm,iTgt1);
      if tflabeled
        obj.assignLabelCoords(lpos);
        obj.enterAccepted(false);
      else
        assert(obj.labeler.hasTrx);

        [tfneighbor,iFrm0,lpos0] = obj.labeler.labelPosLabeledNeighbor(iFrm,iTgt1);
        if tfneighbor
          xy = LabelCore.transformPtsTrx(lpos0,obj.labeler.trx(iTgt1),iFrm0,obj.labeler.trx(iTgt1),iFrm);
        else
          % no neighboring previously labeled points for new target.
          % Just start with current points for previous target.
          
          xy0 = obj.getLabelCoords();
          xy = LabelCore.transformPtsTrx(xy0,obj.labeler.trx(iTgt0),iFrm,obj.labeler.trx(iTgt1),iFrm);
        end
        obj.assignLabelCoords(xy,'tfClip',true);
        obj.enterAdjust(true,false);
      end
      
    end
    
    function clearLabels(obj)
      obj.clearSelected();
      obj.enterAdjust(true,true);
    end
    
    function acceptLabels(obj) 
      obj.enterAccepted(true);
    end
    
    function unAcceptLabels(obj)
      obj.enterAdjust(false,false);
    end 
    
    function axBDF(obj,src,evt) %#ok<INUSD>
      [tf,iSel] = obj.anyPointSelected();
      if tf
        pos = get(obj.hAx,'CurrentPoint');
        pos = pos(1,1:2);
        obj.assignLabelCoordsI(pos,iSel);
        obj.setPointAdjusted(iSel);
        obj.toggleSelectPoint(iSel);
        switch obj.state
          case LabelState.ADJUST
            % none
          case LabelState.ACCEPTED
            obj.enterAdjust(false,false);
        end
      end     
    end
    
    function ptBDF(obj,src,evt) %#ok<INUSD>
      tf = obj.anyPointSelected();
      if tf
        % none
      else
        if obj.state==LabelState.ACCEPTED
          obj.enterAdjust(false,false);
        end
        iPt = get(src,'UserData');
        obj.iPtMove = iPt;
        obj.tfMoved = false;
      end
    end
    
    function wbmf(obj,src,evt) %#ok<INUSD>
      if obj.state==LabelState.ADJUST
        iPt = obj.iPtMove;
        if ~isnan(iPt)
          ax = obj.hAx;
          tmp = get(ax,'CurrentPoint');
          pos = tmp(1,1:2);
          obj.tfMoved = true;
          obj.assignLabelCoordsI(pos,iPt);
          obj.setPointAdjusted(iPt);
        end
      end
    end
    
    function wbuf(obj,src,evt) %#ok<INUSD>
      if obj.state==LabelState.ADJUST
        iPt = obj.iPtMove;
        if ~isnan(iPt) && ~obj.tfMoved
          % point was clicked but not moved
          obj.clearSelected(iPt);
          obj.toggleSelectPoint(iPt);
        end
        
        obj.iPtMove = nan;
        obj.tfMoved = false;
      end
    end
    
    function kpf(obj,src,evt) %#ok<INUSL>
      key = evt.Key;
      modifier = evt.Modifier;      
      tfCtrl = any(strcmp('control',modifier));
      
      switch key
        case {'s' 'space'}
          if obj.state==LabelState.ADJUST
            obj.acceptLabels();
          end
        case {'d' 'equal'}
          obj.labeler.frameUp(tfCtrl);
        case {'a' 'hyphen'}
          obj.labeler.frameDown(tfCtrl);
        case {'leftarrow' 'rightarrow' 'uparrow' 'downarrow'}
          [tfSel,iSel] = obj.anyPointSelected();
          if tfSel
            tfShift = any(strcmp('shift',modifier));
            xy = obj.getLabelCoordsI(iSel);
            switch key
              case 'leftarrow'
                xl = xlim(obj.hAx);
                dx = diff(xl);
                if tfShift
                  xy(1) = xy(1) - dx/obj.DXFACBIG;
                else
                  xy(1) = xy(1) - dx/obj.DXFAC;
                end
                xy(1) = max(xy(1),1);
              case 'rightarrow'
                xl = xlim(obj.hAx);
                dx = diff(xl);
                if tfShift
                  xy(1) = xy(1) + dx/obj.DXFACBIG;
                else
                  xy(1) = xy(1) + dx/obj.DXFAC;
                end
                xy(1) = min(xy(1),obj.labeler.movienc);
              case 'uparrow'
                yl = ylim(obj.hAx);
                dy = diff(yl);
                if tfShift
                  xy(2) = xy(2) - dy/obj.DXFACBIG;
                else
                  xy(2) = xy(2) - dy/obj.DXFAC;
                end
                xy(2) = max(xy(2),1);
              case 'downarrow'
                yl = ylim(obj.hAx);
                dy = diff(yl);
                if tfShift
                  xy(2) = xy(2) + dy/obj.DXFACBIG;
                else
                  xy(2) = xy(2) + dy/obj.DXFAC;
                end
                xy(2) = min(xy(2),obj.labeler.movienr);
            end
            obj.assignLabelCoordsI(xy,iSel);
            switch obj.state
              case LabelState.ADJUST
                obj.setPointAdjusted(iSel);
              case LabelState.ACCEPTED
                obj.enterAdjust(false,false);
            end
          elseif strcmp(key,'leftarrow')
            obj.labeler.frameDown(tfCtrl);
          elseif strcmp(key,'rightarrow')
            obj.labeler.frameUp(tfCtrl);
          end
        case {'0' '1' '2' '3' '4' '5' '6' '7' '8' '9'}
          iPt = str2double(key);
          if iPt==0
            iPt = 10;
          end
          if iPt > obj.nPts
            return;
          end
          obj.clearSelected(iPt);
          obj.toggleSelectPoint(iPt);
      end      
    end
    
    function h = getKeyboardShortcutsHelp(obj) %#ok<MANU>
      h = { ...
        '* A/D, LEFT/RIGHT, or MINUS(-)/EQUAL(=) decrements/increments the frame shown.'
        '* <ctrl>+A, <ctrl>+D, etc decrement and increment by 10 frames.'
        '* S or <space> accepts the labels for the current frame/target.'
        '* 0..9 selects/unselects a point. When a point is selected:'
        '*   LEFT/RIGHT/UP/DOWN adjusts the point.'
        '*   Shift-LEFT, etc adjusts the point by larger steps.' 
        '*   Clicking on the image moves the selected point to that location.'};
    end
    
  end
  
  methods % template
    
    function createTemplate(obj)
      % Initialize "white pts" via user-clicking
      
      assert(false,'Currently not called');
      
      obj.enterAdjust(true);
      
      msg = sprintf('Click to create %d template points.',obj.nPts);
      uiwait(msgbox(msg));
      
      ptsClicked = 0;
      axes(obj.hAx);
      
      while ptsClicked<obj.nPts;
        keydown = waitforbuttonpress;
        if get(0,'CurrentFigure') ~= obj.hFig
          continue;
        end
        if keydown == 0 && strcmpi(get(obj.hFig,'SelectionType'),'normal'),
          tmp = get(obj.hAx,'CurrentPoint');
          xy = tmp(1,1:2);
          iPt = ptsClicked+1;
          LabelCore.assignCoords2Pts(xy,obj.hPts(iPt),obj.hPtsTxt(iPt));
          ptsClicked = iPt;
        elseif keydown == 1 && double(get(obj.hFig,'CurrentCharacter')) == 27,
          % escape
          break;
        end
      end      
    end
    
    function tt = getTemplate(obj)
      % Create a template struct from current pts
      
      tt = struct();
      tt.pts = obj.getLabelCoords();
      lbler = obj.labeler;
      if lbler.hasTrx
        [x,y,th] = lbler.currentTargetLoc();
        tt.loc = [x y];
        tt.theta = th;
      else
        tt.loc = [nan nan];
        tt.theta = nan;
      end
    end
    
    function setTemplate(obj,tt)
      % Set "white points" to template.

      lbler = obj.labeler;      
      tfTemplateHasTarget = ~any(isnan(tt.loc)) && ~isnan(tt.theta);
      tfHasTrx = lbler.hasTrx;
      
      if tfHasTrx && ~tfTemplateHasTarget
        warning('LabelCoreTemplate:template',...
          'Using template saved without target coordinates');
      elseif ~tfHasTrx && tfTemplateHasTarget
        warning('LabelCoreTemplate:template',...
          'Template saved with target coordinates.');
      end
        
      if tfTemplateHasTarget
        [x1,y1,th1] = lbler.currentTargetLoc;
        xys = transformPoints(tt.pts,tt.loc,tt.theta,[x1 y1],th1);
      else        
        xys = tt.pts;
      end
      
      obj.assignLabelCoords(xys,'tfClip',true);
      obj.enterAdjust(true,false);
    end
    
    function setRandomTemplate(obj)
      lbler = obj.labeler;
      [x0,y0] = lbler.currentTargetLoc();
      nr = lbler.movienr;
      nc = lbler.movienc;
      r = round(max(nr,nc)/6);

      n = obj.nPts;
      x = x0 + r*2*(rand(n,1)-0.5);
      y = y0 + r*2*(rand(n,1)-0.5);
      obj.assignLabelCoords([x y],'tfClip',true);
    end
    
  end
  
  methods (Access=private)
    
    function enterAdjust(obj,tfResetPts,tfClearLabeledPos)
      % Enter adjustment state for current frame/tgt.
      %
      % if tfReset, reset all points to pre-adjustment (white).
      % if tfClearLabeledPos, clear labeled pos.
      
      if tfResetPts
        arrayfun(@(x)set(x,'Color',obj.templatePointColor),obj.hPts);
        obj.tfAdjusted(:) = false;
      end
      if tfClearLabeledPos
        obj.labeler.labelPosClear();
      end
        
      obj.iPtMove = nan;
      obj.tfMoved = false;
      
      set(obj.tbAccept,'BackgroundColor',[0.6,0,0],'String','Accept',...
        'Value',0,'Enable','on');
      obj.state = LabelState.ADJUST;
    end
        
    function enterAccepted(obj,tfSetLabelPos)
      % Enter accepted state for current frame/tgt. All points colored. If
      % tfSetLabelPos, all points written to labelpos.
      
      nPts = obj.nPts;
      ptsH = obj.hPts;
      clrs = obj.ptsPlotInfo.Colors;
      for i = 1:nPts
        set(ptsH(i),'Color',clrs(i,:));
      end
      
      obj.tfAdjusted(:) = true;
      obj.clearSelected();
      
      if tfSetLabelPos
        xy = obj.getLabelCoords();
        obj.labeler.labelPosSet(xy);
      end
      set(obj.tbAccept,'BackgroundColor',[0,0.4,0],'String','Accepted',...
        'Value',1,'Enable','on');
      obj.state = LabelState.ACCEPTED;
    end
    
    function setPointAdjusted(obj,iSel)
      if ~obj.tfAdjusted(iSel)
        obj.tfAdjusted(iSel) = true;
        set(obj.hPts(iSel),'Color',obj.ptsPlotInfo.Colors(iSel,:));
      end
    end

    function toggleSelectPoint(obj,iPt)
      tfSel = ~obj.tfPtSel(iPt);
      obj.tfPtSel(:) = false;
      obj.tfPtSel(iPt) = tfSel;
      
      if tfSel
        set(obj.hPts(iPt),'Marker',obj.selectedPointMarker);
      else
        set(obj.hPts(iPt),'Marker',obj.ptsPlotInfo.Marker);
      end
    end
    
    function [tf,iSelected] = anyPointSelected(obj)
      tf = any(obj.tfPtSel);
      iSelected = find(obj.tfPtSel,1);
    end
    
    function clearSelected(obj,iExclude)
      tf = obj.tfPtSel;
      if exist('iExclude','var')>0
        tf(iExclude) = false;
      end
      iSel = find(tf);
      for i = iSel(:)'
        obj.toggleSelectPoint(i);
      end
    end
        
  end
  
end