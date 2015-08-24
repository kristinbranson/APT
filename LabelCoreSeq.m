classdef LabelCoreSeq < LabelCore   
  % Label mode 1 (Sequential)
  %
  % There are three labeling states: 'label', 'adjust', 'accepted'.
  %
  % During the labeling state, points are being clicked in order. This
  % includes the state where there are zero points clicked (fresh image).
  %
  % During the adjustment state, points may be adjusted by click-dragging.
  %
  % When any/all adjustment is complete, tbAccept is clicked and we enter
  % the accepted stage. This locks the labeled points for this frame and
  % writes to .labeledpos.
  %
  % pbClear is enabled at all times. Clicking it returns to the 'label'
  % state and clears any labeled points.
  %
  % tbAccept is disabled during 'label'. During 'adjust', its name is
  % "Accept" and clicking it moves to the 'accepted' state. During
  % 'accepted, its name is "Adjust" and clicking it moves to the 'adjust'
  % state.
  %
  % When multiple targets are present, all actions/transitions are for
  % the current target. Acceptance writes to .labeledpos for the current
  % target. Changing targets is like changing frames; all pre-acceptance
  % actions are discarded.
      
  properties
    iPtMove;
    nPtsLabeled; % scalar integer. 0..nPts, or inf.
  end
  
  methods
    
    function obj = LabelCoreSeq(varargin)
      obj = obj@LabelCore(varargin{:});
    end
    
    function newFrame(obj,iFrm0,iFrm1,iTgt) %#ok<INUSL>
      obj.newFrameOrTarget(iFrm1,iTgt);
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm) %#ok<INUSL>
      obj.newFrameOrTarget(iFrm,iTgt1);
    end
    
    function clearLabels(obj)
      obj.beginLabel();
    end
    
    function acceptLabels(obj)
      obj.beginAccepted(true);
    end
    
    function unAcceptLabels(obj)
      obj.beginAdjust();
    end
    
    function axBDF(obj,~,~)
      if obj.state==LabelState.LABEL
        ax = obj.hAx;
        
        nlbled = obj.nPtsLabeled;
        if nlbled>=obj.nPts
          assert(false); % adjustment mode only
        else % 0..nPts-1
          tmp = get(ax,'CurrentPoint');
          x = tmp(1,1);
          y = tmp(1,2);
          
          i = nlbled+1;
          set(obj.hPts(i),'XData',x,'YData',y);
          set(obj.hPtsTxt(i),'Position',[x+obj.DT2P y+obj.DT2P]);
          obj.nPtsLabeled = i;
          
          if i==obj.nPts
            obj.beginAdjust();
          end
        end
      end
    end
    
    function ptBDF(obj,src,~)
      switch obj.state
        case LabelState.ADJUST
          obj.iPtMove = get(src,'UserData');
        case LabelState.ACCEPTED
          obj.beginAdjust();
          obj.iPtMove = get(src,'UserData');
      end
    end
    
    function wbmf(obj,~,~)
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
    
    function wbuf(obj,~,~)
      if obj.state==LabelState.ADJUST
        obj.iPtMove = nan;
      end
    end
    
    function kpf(obj,~,evt)
      key = evt.Key;
      modifier = evt.Modifier;
      
      tfCtrl = ismember('control',modifier);
      switch key
        case {'s' 'space'} % accept
          if obj.state==LabelState.ADJUST
            obj.acceptLabels();
          end
        case {'rightarrow' 'd' 'equal'}
          obj.labeler.frameUp(tfCtrl);
        case {'leftarrow' 'a' 'hyphen'}
          obj.labeler.frameDown(tfCtrl);
      end
    end
    
    function h = getKeyboardShortcutsHelp(obj) %#ok<MANU>
      h = { ...
        '* A/D, LEFT/RIGHT, or MINUS(-)/EQUAL(=) decrement/increment the frame shown.'
        '* <ctrl>+A and <ctrl>+D decrement and increment by 10 frames.'
        '* S or <space> accepts the labels for the current frame/target.'};
    end
          
  end
  
  methods
    
    function newFrameOrTarget(obj,iFrm,iTgt)
      % React to new frame or target. Set mode1 label state (.lbl1_*) 
      % according to labelpos. If a frame is not labeled, then start fresh 
      % in Label state. Otherwise, start in Accepted state with saved labels.
            
      [tflabeled,lpos] = obj.labeler.labelPosIsLabeled(iFrm,iTgt);
      if tflabeled
        obj.nPtsLabeled = obj.nPts;
        obj.assignLabelCoords(lpos);
        obj.iPtMove = nan;
        obj.beginAccepted(false); % I guess could just call with true arg
      else
        obj.beginLabel();
      end
    end
    
    function beginLabel(obj)
      % Enter Label state and clear all mode1 label state for current
      % frame/target
      
      set(obj.tbAccept,'BackgroundColor',[0.4 0.0 0.0],...
        'String','','Enable','off','Value',0);
      
      obj.nPtsLabeled = 0;
      arrayfun(@(x)set(x,'Xdata',nan,'ydata',nan),obj.hPts);
      arrayfun(@(x)set(x,'Position',[nan nan 1],'hittest','off'),obj.hPtsTxt);
      obj.iPtMove = nan;
      obj.labeler.labelPosClear();
      
      obj.state = LabelState.LABEL;      
    end
       
    function beginAdjust(obj)
      % Enter adjustment state for current frame/target
      
      assert(obj.nPtsLabeled==obj.nPts);
            
      obj.iPtMove = nan;
      
      set(obj.tbAccept,'BackgroundColor',[0.6,0,0],'String','Accept',...
        'Value',0,'Enable','on');
      obj.state = LabelState.ADJUST;
    end
    
    function beginAccepted(obj,tfSetLabelPos)
      % Enter accepted state (for current frame)
      
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