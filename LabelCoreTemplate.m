classdef LabelCoreTemplate < LabelCore
  
  % DESCRIPTION
  % In Template mode, there is a set of template/"white" points on the
  % image at all times. (When starting, these points need to be either
  % specified or loaded/imported.) To label a frame, adjust the points as 
  % necessary and accept. 
  %
  %
  % LABEL MODE 2 IMPL NOTES
  % 2 States:
  % - Adjusting
  % -- 'untouched' template points are white
  % -- 'touched' template points are colored
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
    iPtMove;
    tfAdjusted;                  % nLabelPoints x 1 logical vec. If true, pt has been adjusted from template
    
    %template;                    % LabelTemplate obj. "original" template
    templatePtsColor = [1 1 1];  % 1 x 3 RGB
  end
  
  methods
    
    function obj = LabelCoreTemplate(varargin)
      obj = obj@LabelCore(varargin{:});
    end
    
    function initHook(obj)
      obj.setRandomTemplate();
      obj.tfAdjusted = false(obj.nPts,1);
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
          obj.assignLabelCoords(xy);
        else
          % none, leave pts as-is
        end          
        obj.enterAdjust(true);
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
        obj.assignLabelCoords(xy);
        obj.enterAdjust(true);
      end
      
    end
    
    function clearLabels(obj)
      obj.enterAdjust(true);
    end
    
    function acceptLabels(obj) 
      obj.enterAccepted(true);
    end
    
    function axBDF(obj,src,evt) %#ok<INUSD>
      % if in Adjust:pt-selected state, then clicking relocates pt
    end
    
    function ptBDF(obj,src,evt) %#ok<INUSD>
      iPt = get(src,'UserData');
      obj.iPtMove = iPt;
      switch obj.state
        case LabelState.ADJUST
          if ~obj.tfAdjusted(iPt)
            obj.tfAdjusted(iPt) = true;
            set(src,'Color',obj.ptColors(iPt,:));
          end
        case LabelState.ACCEPTED
          obj.enterAdjust(false);
          if ~obj.tfAdjusted(iPt)
            obj.tfAdjusted(iPt) = true;
            set(src,'Color',obj.ptColors(iPt,:));
          end
      end
    end
    
    function wbmf(obj,src,evt) %#ok<INUSD>
      if obj.state==LabelState.ADJUST
        iPt = obj.iPtMove;
        if ~isnan(iPt) % should always be true
          ax = obj.hAx;
          tmp = get(ax,'CurrentPoint');
          pos = tmp(1,1:2);
          set(obj.hPts(iPt),'XData',pos(1),'YData',pos(2));
          pos(1) = pos(1) + obj.DT2P;
          set(obj.hPtsTxt(iPt),'Position',pos);
        end
      end
    end
    
    function wbuf(obj,src,evt) %#ok<INUSD>
      if obj.state==LabelState.ADJUST
        obj.iPtMove = nan;
      end
    end
    
    function kpf(obj,src,evt) %#ok<INUSD>
    end
    
    function getKeyboardShortcutsHelp(obj) %#ok<MANU>
    end
    
  end
  
  methods % template
    
    function createTemplate(obj)
      % Initialize "white pts" via user-clicking
      
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
      
      obj.assignLabelCoords(xys);
      obj.enterAdjust(true);
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
      obj.assignLabelCoords([x y]);
    end
    
  end
  
  methods (Access=private)
    
    function enterAdjust(obj,tfReset)
      % Enter adjustment state for current frame/tgt.
      %
      % if tfReset, reset all points to pre-adjustment
      
      if tfReset
        arrayfun(@(x)set(x,'Color',obj.templatePtsColor),obj.hPts);
        obj.tfAdjusted(:) = false;
      end
      obj.iPtMove = nan;
      
      set(obj.tbAccept,'BackgroundColor',[0.6,0,0],'String','Accept',...
        'Value',0,'Enable','on');
      obj.state = LabelState.ADJUST;
    end
        
    function enterAccepted(obj,tfSetLabelPos)
      % Enter accepted state for current frame/tgt. All points colored. If
      % tfSetLabelPos, all points written to labelpos.
      
      nPts = obj.nPts;
      ptsH = obj.hPts;
      clrs = obj.ptColors;
      for i = 1:nPts
        set(ptsH(i),'Color',clrs(i,:));
      end
      
      obj.tfAdjusted(:) = true;
      
      if tfSetLabelPos
        xy = obj.getLabelCoords();
        obj.labeler.labelPosSet(xy);
      end
      set(obj.tbAccept,'BackgroundColor',[0,0.4,0],'String','Accepted',...
        'Value',1,'Enable','on');
      obj.state = LabelState.ACCEPTED;
    end
        
  end
  
end