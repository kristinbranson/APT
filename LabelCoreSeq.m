classdef LabelCoreSeq < LabelCore   
  % Label mode 1 (Sequential)
  %
  % There are three labeling states: 'label', 'adjust', 'accepted'.
  %
  % During the labeling state, points are be ing clicked in order. This
  % includes the state where there are zero points clicked (fresh image).
  %
  % Once all points have been clicked, the adjustment state is entered.
  % Points may be adjusted by click-dragging or using hotkeys as in
  % Template Mode.
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
  %
  % Occluded. In the 'label' state, clicking in the full-occluded subaxis
  % sets the current point to be fully occluded. 
  
  properties
    supportsMultiView = false;
    supportsCalibration = false;
  end
        
  properties
    iPtMove; % scalar. Either nan, or index of pt being moved
    nPtsLabeled; % scalar integer. 0..nPts, or inf.

    % Templatemode-like behavior in 'adjust' and 'accepted' stages
    tfPtSel; % [nPtsx1] logical vec. If true, pt is currently selected.
    kpfIPtFor1Key; % scalar positive integer. This is the point index that
                   % the '1' hotkey maps to, eg typically this will take the 
                   % values 1, 11, 21, ...
  end
  
  methods
    function set.kpfIPtFor1Key(obj,val)
      obj.kpfIPtFor1Key = val;
      obj.refreshTxLabelCoreAux();
    end
  end
  
  methods
    
    function obj = LabelCoreSeq(varargin)
      obj = obj@LabelCore(varargin{:});
    end
    
    function initHook(obj)
      obj.tfPtSel = false(obj.nPts,1);
      obj.txLblCoreAux.Visible = 'on';
      obj.kpfIPtFor1Key = 1;
      obj.refreshTxLabelCoreAux();
    end
    
    function newFrame(obj,iFrm0,iFrm1,iTgt) %#ok<INUSL>
      obj.newFrameTarget(iFrm1,iTgt);
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm) %#ok<INUSL>
      obj.newFrameTarget(iFrm,iTgt1);
    end
    
    function newFrameAndTarget(obj,~,iFrm1,~,iTgt1)
      obj.newFrameTarget(iFrm1,iTgt1);
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
      obj.axOrAxOccBDF(false);
    end
    
    function axOccBDF(obj,~,~)
      obj.axOrAxOccBDF(true);
    end
   
    function axOrAxOccBDF(obj,tfAxOcc)
      switch obj.state
        case LabelState.LABEL
          ax = obj.hAx;
          
          nlbled = obj.nPtsLabeled;
          assert(nlbled<obj.nPts);
          i = nlbled+1;
          if tfAxOcc
            obj.tfOcc(i) = true;
            obj.refreshOccludedPts();
          else
            tmp = get(ax,'CurrentPoint');
            x = tmp(1,1);
            y = tmp(1,2);
            obj.assignLabelCoordsIRaw([x y],i);
          end
          obj.nPtsLabeled = i;
          if i==obj.nPts
            obj.beginAdjust();
          end
        case {LabelState.ADJUST LabelState.ACCEPTED}
          [tf,iSel] = obj.anyPointSelected();
          if tf
            pos = get(obj.hAx,'CurrentPoint');
            pos = pos(1,1:2);
            obj.assignLabelCoordsIRaw(pos,iSel);
            obj.toggleSelectPoint(iSel);
%             if obj.tfOcc(iSel)
%               obj.tfOcc(iSel) = false;
%               obj.refreshOccludedPts();
%             end
            % estOcc status unchanged
            if obj.state==LabelState.ACCEPTED
              obj.beginAdjust();
            end
          end
      end
    end
    
    function undoLastLabel(obj)
      switch obj.state
        case {LabelState.LABEL LabelState.ADJUST}
          nlbled = obj.nPtsLabeled;
          if nlbled>0
            if obj.tfOcc(nlbled)
              obj.tfOcc(nlbled) = false;
              obj.refreshOccludedPts();
            else
              obj.assignLabelCoordsIRaw([nan nan],nlbled);
            end
            obj.nPtsLabeled = nlbled-1;
            
            if obj.state==LabelState.ADJUST
              assert(nlbled==obj.nPts);
              obj.adjust2Label();
            end
          end          
      end
    end
        
    function ptBDF(obj,src,~)
      tf = obj.anyPointSelected();
      if tf
        % none
      else
        switch obj.state
          case {LabelState.ADJUST LabelState.ACCEPTED}          
            iPt = get(src,'UserData');
            if obj.state==LabelState.ACCEPTED
              obj.beginAdjust();
            end
            obj.iPtMove = iPt;
        end
      end
    end
    
    function wbmf(obj,~,~)
      if obj.state==LabelState.ADJUST
        iPt = obj.iPtMove;
        if ~isnan(iPt)
          ax = obj.hAx;
          tmp = get(ax,'CurrentPoint');
          pos = tmp(1,1:2);          
          obj.assignLabelCoordsIRaw(pos,iPt);
        end
      end
    end
    
    function wbuf(obj,~,~)
      if obj.state==LabelState.ADJUST
        obj.iPtMove = nan;
      end
    end
    
    function tfKPused = kpf(obj,~,evt)
      key = evt.Key;
      modifier = evt.Modifier;
      tfCtrl = ismember('control',modifier);
      tfShft = any(strcmp('shift',modifier));

      tfKPused = true;
      lObj = obj.labeler;
      if strcmp(key,'z') && tfCtrl
        obj.undoLastLabel();
      elseif strcmp(key,'h') && tfCtrl
        obj.labelsHideToggle();
      elseif any(strcmp(key,{'s' 'space'})) && ~tfCtrl % accept
        if obj.state==LabelState.ADJUST
          obj.acceptLabels();
        end
      elseif any(strcmp(key,{'d' 'equal'}))
        lObj.frameUp(tfCtrl);
      elseif any(strcmp(key,{'a' 'hyphen'}))
        lObj.frameDown(tfCtrl);
      elseif any(strcmp(key,{'leftarrow' 'rightarrow' 'uparrow' 'downarrow'}))
        [tfSel,iSel] = obj.anyPointSelected();
        if tfSel % && ~obj.tfOcc(iSel)
          tfShift = any(strcmp('shift',modifier));
          xy = obj.getLabelCoordsI(iSel);
          switch key
            case 'leftarrow'
              dxdy = -lObj.videoCurrentRightVec();
            case 'rightarrow'
              dxdy = lObj.videoCurrentRightVec();
            case 'uparrow'
              dxdy = lObj.videoCurrentUpVec();
            case 'downarrow'
              dxdy = -lObj.videoCurrentUpVec();
          end
          if tfShift
            xy = xy + dxdy*10;
          else
            xy = xy + dxdy;
          end
          xy = lObj.videoClipToVideo(xy);
          obj.assignLabelCoordsIRaw(xy,iSel);
          switch obj.state
            case LabelState.ADJUST
              % none
            case LabelState.ACCEPTED              
              obj.beginAdjust();
          end
        elseif strcmp(key,'leftarrow')
          if tfShft
            lObj.frameUpNextLbled(true);
          else
            lObj.frameDown(tfCtrl);
          end
        elseif strcmp(key,'rightarrow')
          if tfShft
            lObj.frameUpNextLbled(false);
          else
            lObj.frameUp(tfCtrl);
          end
        else
          % should never occur
          tfKPused = false;
        end
      elseif strcmp(key,'backquote')
        iPt = obj.kpfIPtFor1Key+10;
        if iPt>obj.nPts
          iPt = 1;
        end
        obj.kpfIPtFor1Key = iPt;
      elseif any(strcmp(key,{'0' '1' '2' '3' '4' '5' '6' '7' '8' '9'}))
        iPt = str2double(key);
        if iPt==0
          iPt = 10;
        end
        iPt = iPt+obj.kpfIPtFor1Key-1;
        if iPt>obj.nPts
          return;
        end
        obj.clearSelected(iPt);
        obj.toggleSelectPoint(iPt);
      else
        tfKPused = false;
      end
    end
    
    function h = getLabelingHelp(obj) %#ok<MANU>
      h = { ...
        '* A/D, LEFT/RIGHT, or MINUS(-)/EQUAL(=) decrement/increment the frame shown.'
        '* <ctrl>+A/D, LEFT/RIGHT etc decrement/increment by 10 frames.'
        '* S or <space> accepts the labels for the current frame/target.'};
    end
    
  end
  
  methods
    
    function newFrameTarget(obj,iFrm,iTgt)
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
      obj.assignLabelCoords(nan(obj.nPts,2));
      obj.nPtsLabeled = 0;
      obj.iPtMove = nan;
      obj.labeler.labelPosClear();      
      obj.state = LabelState.LABEL;      
    end
    
    function adjust2Label(obj)
      % enter LABEL from ADJUST
      set(obj.tbAccept,'BackgroundColor',[0.4 0.0 0.0],...
        'String','','Enable','off','Value',0);      
      obj.iPtMove = nan;
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
      obj.clearSelected();
      set(obj.tbAccept,'BackgroundColor',[0,0.4,0],'String','Accepted',...
        'Value',1,'Enable','on');
      obj.state = LabelState.ACCEPTED;
    end
     
    % C+P
    function [tf,iSelected] = anyPointSelected(obj)
      tf = any(obj.tfPtSel);
      iSelected = find(obj.tfPtSel,1);
    end
    
    %C+P
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
    
    % C+P
    function toggleSelectPoint(obj,iPt)
      tfSel = ~obj.tfPtSel(iPt);
      obj.tfPtSel(:) = false;
      obj.tfPtSel(iPt) = tfSel;

      obj.refreshPtMarkers(iPt);
      
%       % Also update hPtsOcc markers
%       if tfSel
%         mrkr = obj.ptsPlotInfo.TemplateMode.SelectedPointMarker;
%       else
%         mrkr = obj.ptsPlotInfo.Marker;
%       end
%       set(obj.hPtsOcc(iPt),'Marker',mrkr);
    end
    
    % C+P
    function refreshPtMarkers(obj,iPts)
      % Update obj.hPts Markers based on .tfEstOcc and .tfPtSel.
      
      ppi = obj.ptsPlotInfo;
      ppitm = ppi.TemplateMode;

      hPoints = obj.hPts(iPts);
      tfSel = obj.tfPtSel(iPts);
      %tfEO = obj.tfEstOcc(iPts);
      
      %set(hPoints(tfSel & tfEO),'Marker',ppitm.SelectedOccludedMarker); % historical quirk, use props instead of ppi; fix this at some pt
      set(hPoints(tfSel),'Marker',ppitm.SelectedPointMarker);
      %set(hPoints(~tfSel & tfEO),'Marker',ppi.OccludedMarker);
      set(hPoints(~tfSel),'Marker',ppi.Marker);
    end
    
    % C+P
    function refreshTxLabelCoreAux(obj)
      iPt0 = obj.kpfIPtFor1Key;
      iPt1 = iPt0+9;
      str = sprintf('Hotkeys 0-9 map to points %d-%d',iPt0,iPt1);
      obj.txLblCoreAux.String = str;      
    end

  end
  
end