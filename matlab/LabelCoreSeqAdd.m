classdef LabelCoreSeqAdd < LabelCoreSeq
% Sequential adding of new landmarks
  
  % Label mode 1 (Sequential)
  %
  % There are two labeling states: 'label' and 'accepted'.
  %
  % During the labeling state, points are being clicked in order. This
  % includes the state where there are zero points clicked (fresh image).
  %
  % Once all points have been clicked, the accepted state is entered.
  % This writes to .labeledpos. Points may be adjusted by click-dragging or
  % using hotkeys as in Template Mode. 
  %
  % pbClear is enabled at all times. Clicking it returns to the 'label'
  % state and clears any labeled points.
  %
  % tbAccept is disabled at all times. During 'accepted', its name is
  % green and its name is "Labeled" and during 'label' its name is
  % "Unlabeled" and it is red. 
  %
  % When multiple targets are present, all actions/transitions are for
  % the current target. Acceptance writes to .labeledpos for the current
  % target. Changing targets is like changing frames; all pre-acceptance
  % actions are discarded.
  %
  % Occluded. In the 'label' state, clicking in the full-occluded subaxis
  % sets the current point to be fully occluded. 
          
  properties
    nadd; % number of landmarks being added
    nold; % number of landmarks to leave frozen

    % Templatemode-like behavior in 'adjust' and 'accepted' stages
    % moved to parent class
    %kpfIPtFor1Key; % scalar positive integer. This is the point index that
                   % the '1' hotkey maps to, eg typically this will take the 
                   % values 1, 11, 21, ...
  end
    
  properties
    unsupportedKPFFns = {} ;  % cell array of field names for objects that have general keypressfcn 
                              % callbacks but are not supported for this LabelCore
  end
  
  methods
    
    function obj = LabelCoreSeqAdd(varargin)
      obj = obj@LabelCoreSeq(varargin{:});
      set(obj.tbAccept,'Enable','off','Style','pushbutton');
    end
    
    function initHook(obj)
      obj.nadd = obj.labeler.nLabelPointsAdd;
      obj.nold = obj.nPts-obj.nadd;
      obj.txLblCoreAux.Visible = 'on';
      obj.kpfIPtFor1Key = obj.nold+1;
      obj.refreshTxLabelCoreAux();
      
      % points that were labeled in the past should not be selectable,
      % movable
      arrayfun(@(x)set(x,'HitTest','off','ButtonDownFcn',''),obj.hPts(1:obj.nold));
      if ~isempty(obj.hPtsOcc),
        arrayfun(@(x)set(x,'HitTest','off','ButtonDownFcn',''),obj.hPtsOcc(1:obj.nold));
      end
      
      % AL 20190203 semi-hack. init to something/anything to avoid error 
      % with .state unset. Did the same with LabelCoreTemplate. A little
      % disturbing, something has changed with labelCore init or
      % Labeler.labelingInit but not clear what. Prob not a super-huge risk
      % low prob of significant data loss
      obj.state = LabelState.ADJUST; 
    end
    
    function newFrame(obj,iFrm0,iFrm1,iTgt,tfForceUpdate) %#ok<INUSL>
      if nargin < 5
        tfForceUpdate = false;
      end
      obj.newFrameTarget(iFrm1,iTgt);
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm) %#ok<INUSL>
      obj.newFrameTarget(iFrm,iTgt1);
    end
    
    function newFrameAndTarget(obj,~,iFrm1,~,iTgt1,tfForceUpdate)
      if nargin < 6
        tfForceUpdate = false;
      end
      obj.newFrameTarget(iFrm1,iTgt1);
    end
    
    function clearLabels(obj)
      obj.beginAdd([],true);
    end
    
    function acceptLabels(obj)
      obj.beginAccepted(true);
      obj.labeler.InitializePrevAxesTemplate();
    end
    
    function unAcceptLabels(obj)
      % go to the next frame to label
      if ~isempty(obj.nexttbl),
        obj.goToNextTable();
      else
        obj.goToNextUnlabeled();
      end
    end

    % setNextTable(obj,tbl)
    % Set a table of frames, targets, and movies to be labeled. When you
    % click the Next button, it will take you to the next unlabeled frame,
    % target and movie from this table. 
    % Input: 
    % tbl: MATLAB table, each row of which corresponds to a frame to label.
    % It should have the following variables:
    % 'mov': Full path of the movie you want to label
    % 'iTgt': Index of the target you want to label
    % 'frm': Frame number you want to label
    function setNextTable(obj,tbl)
      setNextTable@LabelCore(obj,tbl);
      obj.goToNextTable();
    end

    function goToNextTable(obj)

      % find what is left to label
      labels = obj.labeler.labelsGTaware();
      todo = false(size(obj.nexttbl,1),1);
      % this assumes single view
      [movnames,~,rowidx] = unique(obj.nexttbl.mov);
      for i = 1:numel(movnames),
        idxcurr = rowidx == i;
        [tf,mov,~,~] = obj.labeler.movieSetInProj(movnames{i});
        if ~tf,
          errordlg(sprintf('Movie %s is not in this project',movnames{i}),'Bad next table');
          return;
        end
        [frms,tgts] = Labels.isPartiallyLabeledT(labels{mov},nan,obj.nold);
        todo(idxcurr) = ismember([obj.nexttbl.frm(idxcurr),obj.nexttbl.iTgt(idxcurr)],[frms(:),tgts(:)],'rows');
      end

      % next unlabeled in our sequence
      nextj = find(todo(obj.nexti+1:end),1);
      if isempty(nextj),
        nextj = find(todo,1);
        if isempty(nextj),
          msgbox('No partially labeled frames found within the table! You might be done!','Done adding landmarks','replace');
          return;
        else
          obj.nexti = nextj;
        end
      else
        obj.nexti = obj.nexti + nextj;
      end

      frm = obj.nexttbl.frm(obj.nexti);
      if ismember('iTgt',obj.nexttbl.Properties.VariableNames),
        tgt = obj.nexttbl.iTgt(obj.nexti);
      end
      movname = obj.nexttbl.mov{obj.nexti};
      [tf,mov,~,~] = obj.labeler.movieSetInProj(movname);
      if ~tf,
        errordlg(sprintf('Movie %s is not in this project',movname),'Bad next table');
        return;
      end
      if obj.labeler.currMovie ~= mov,
        obj.labeler.movieSetGUI(mov);
      end
      if obj.labeler.currTarget ~= tgt,
        obj.labeler.setFrameAndTargetGUI(frm,tgt);
      else
        obj.labeler.setFrameGUI(frm);
      end

    end

    function goToNextUnlabeled(obj)
      [frm,tgt,mov] = obj.findUnlabeled();
      if isempty(frm),
        msgbox('No partially labeled frames found! You might be done!','Done adding landmarks','replace');
        return;
      end
      if obj.labeler.currMovie ~= mov,
        obj.labeler.movieSetGUI(mov);
      end
      if obj.labeler.currTarget ~= tgt,
        obj.labeler.setFrameAndTargetGUI(frm,tgt);
      else
        obj.labeler.setFrameGUI(frm);
      end
    end
    
    function [frm,tgt,mov] = findUnlabeled(obj)
      labels = obj.labeler.labelsGTaware();
      currtgt = obj.labeler.currTarget;
      currmov = obj.labeler.currMovie;
      currfrm = obj.labeler.currFrame;
      frms = Labels.isPartiallyLabeledT(labels{currmov},currtgt,obj.nold);
      if ~isempty(frms),
        i = find(frms > currfrm,1);
        if isempty(i),
          frm = frms(1);
        else
          frm = frms(i);
        end
        tgt = currtgt;
        mov = currmov;
        return;
      end
      [frms,tgts] = Labels.isPartiallyLabeledT(labels{currmov},nan,obj.nold);
      if ~isempty(frms),
        frm = frms(1);
        tgt = tgts(1);
        mov = currmov;
        return;
      end
      for mov = 1:size(labels,1),
        [frms,tgts] = Labels.isPartiallyLabeledT(labels{mov},nan,obj.nold);
        if ~isempty(frms),
          frm = frms(1);
          tgt = tgts(1);
          return;
        end
      end
      frm = [];
      tgt = [];
      mov = [];
    end
    

    
    function axBDF(obj,src,evt) 
      if ~obj.labeler.isReady || evt.Button>1
        return;
      end
      
      if obj.isPanZoom(),
        return;
      end

      mod = obj.hFig.CurrentModifier;
      tfShift = any(strcmp(mod,'shift'));
      switch obj.state
        case LabelState.LABEL
          obj.hlpAxBDFLabelState(false,tfShift);
        case {LabelState.ADJUST LabelState.ACCEPTED}
          [tf,iSel] = obj.anyPointSelected();
          if tf
            pos = get(obj.hAx,'CurrentPoint');
            pos = pos(1,1:2);
            obj.assignLabelCoordsIRaw(pos,iSel);
            obj.labeler.labelPosSetI(pos,iSel);
            obj.tfEstOcc(iSel) = tfShift; % following toggleSelectPoint call will call refreshPtMarkers
            obj.toggleSelectPoint(iSel);
            if obj.tfOcc(iSel)
              obj.tfOcc(iSel) = false;
              obj.refreshOccludedPts();
            end
            % estOcc status unchanged
            if obj.state==LabelState.ACCEPTED
              % KB 20181029: removing adjust state
              %obj.beginAdjust();
            end
          end
      end
    end
    
    function axOccBDF(obj,~,~)
      
      if ~obj.labeler.isReady,
        return;
      end
      if obj.isPanZoom(),
        return;
      end

      mod = obj.hFig.CurrentModifier;
      tfShift = any(strcmp(mod,'shift'));

      switch obj.state
        case LabelState.LABEL
          obj.hlpAxBDFLabelState(true,tfShift);
        case {LabelState.ADJUST LabelState.ACCEPTED}
          [tf,iSel] = obj.anyPointSelected();
          if tf
            if obj.tfEstOcc(iSel)
              obj.tfEstOcc(iSel) = false; 
              % following toggleSelectPoint call will call refreshPtMarkers
            else
              obj.labeler.labelPosSetIFullyOcc(iSel);
            end
            obj.toggleSelectPoint(iSel);        
            obj.tfOcc(iSel) = true;
            obj.refreshOccludedPts();
            % estOcc status unchanged
            if obj.state==LabelState.ACCEPTED
              % KB 20181029: removing adjust state
              %obj.beginAdjust();
            end
          end
      end
    end
       
    function hlpAxBDFLabelState(obj,tfAxOcc,tfShift)
      
      % BDF in LabelState.LABEL. .tfOcc, .tfEstOcc, .tfSel start off as all
      % false in beginAdd();
      
      nlbled = obj.nPtsLabeled;
      assert(nlbled<obj.nPts);
      i = nlbled+1;
      if tfAxOcc
        obj.tfOcc(i) = true;
        obj.refreshOccludedPts();
      else
        ax = obj.hAx;
        tmp = get(ax,'CurrentPoint');
        x = tmp(1,1);
        y = tmp(1,2);
        obj.assignLabelCoordsIRaw([x y],i);
        if tfShift
          obj.tfEstOcc(i) = true;
        end
        obj.refreshPtMarkers('iPts',i);
      end
      obj.nPtsLabeled = i;
      if i==obj.nPts
        % KB 2018029: removing adjust mode
        %obj.beginAdjust();
        obj.acceptLabels();
      end
    end
    
    function undoLastLabel(obj)
      switch obj.state
        case {LabelState.LABEL LabelState.ADJUST}
          nlbled = obj.nPtsLabeled;
          if nlbled>0
            obj.tfSel(nlbled) = false;
            obj.tfEstOcc(nlbled) = false;
            obj.tfOcc(nlbled) = false;
            obj.refreshOccludedPts();
            obj.refreshPtMarkers('iPts',nlbled,'doPtsOcc',true);
            obj.assignLabelCoordsIRaw([nan nan],nlbled);
            obj.nPtsLabeled = nlbled-1;

            % KB 20181029: removing adjust state
%             if obj.state==LabelState.ADJUST
%               assert(nlbled==obj.nPts);
%               obj.adjust2Label();
%             end
          end          
      end
    end
        
    function ptBDF(obj,src,evt)
      if ~obj.labeler.isReady || evt.Button>1
        return;
      end
      if obj.isPanZoom(),
        return;
      end

      switch obj.state
        case {LabelState.ADJUST LabelState.ACCEPTED}
          iPt = get(src,'UserData');
          obj.toggleSelectPoint(iPt);
          % KB 20181029: removing adjust state
          %             if obj.state==LabelState.ACCEPTED
          %               obj.beginAdjust();
          %             end
          obj.iPtMove = iPt;
      end
    end
    
    function wbmf(obj,~,~)
      % KB 20181029: removing adjust state
      if isempty(obj.state) || ~obj.labeler.isReady,
        return;
      end
      if obj.state == LabelState.ADJUST || obj.state == LabelState.ACCEPTED,
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
      
      if ~obj.labeler.isReady,
        return;
      end
      
      % KB 20181029: removing adjust state
      if ismember(gco,obj.labeler.controller_.tvTrx_.hTrx),
        return;
      end
      if obj.state == LabelState.ADJUST || obj.state == LabelState.ACCEPTED && ~isempty(obj.iPtMove) && ~isnan(obj.iPtMove),
        obj.iPtMove = nan;
        obj.storeLabels();
      end
    end
    
    function tfKPused = kpf(obj,~,evt)
      
      if ~obj.labeler.isReady,
        return;
      end
      
      key = evt.Key;
      modifier = evt.Modifier;
      tfCtrl = ismember('control',modifier);
      tfShft = any(strcmp('shift',modifier));

      tfKPused = true;
      lObj = obj.labeler;
      if strcmp(key,'z') && tfCtrl
        obj.undoLastLabel();
      elseif strcmp(key,'o') && ~tfCtrl
        [tfSel,iSel] = obj.anyPointSelected();
        if tfSel
          obj.toggleEstOccPoint(iSel);
        end
        if obj.state == LabelState.ACCEPTED,
          obj.storeLabels();
        end
        % KB 20181029: removing adjust state
%       elseif any(strcmp(key,{'s' 'space'})) && ~tfCtrl % accept
%         if obj.state==LabelState.ADJUST
%           obj.acceptLabels();
%         end
      elseif any(strcmp(key,{'d' 'equal'}))
        lObj.frameUpGUI(tfCtrl);
      elseif any(strcmp(key,{'a' 'hyphen'}))
        lObj.frameDownGUI(tfCtrl);
      elseif any(strcmp(key,{'leftarrow' 'rightarrow' 'uparrow' 'downarrow'}))
        [tfSel,iSel] = obj.anyPointSelected();
        if tfSel % && ~obj.tfOcc(iSel)
          tfShift = any(strcmp('shift',modifier));
          xy = obj.getLabelCoordsI(iSel);
          switch key
            case 'leftarrow'
              dxdy = -lObj.controller_.videoCurrentRightVec();
            case 'rightarrow'
              dxdy = lObj.controller_.videoCurrentRightVec();
            case 'uparrow'
              dxdy = lObj.controller_.videoCurrentUpVec();
            case 'downarrow'
              dxdy = -lObj.controller_.videoCurrentUpVec();
          end
          if tfShift
            xy = xy + dxdy*10;
          else
            xy = xy + dxdy;
          end
          xy = obj.controller.videoClipToVideo(xy);
          obj.assignLabelCoordsIRaw(xy,iSel);
          switch obj.state
            case LabelState.ADJUST
              % none
            case LabelState.ACCEPTED              
              % KB 20181029: removing adjust state
              %obj.beginAdjust();
          end
        else
          tfKPused = false;
        end
      elseif strcmp(key,'backquote')
        iPt = obj.kpfIPtFor1Key+10;
        if iPt>obj.nPts
          iPt = obj.nold+1;
        end
        obj.kpfIPtFor1Key = iPt;
      elseif any(strcmp(key,{'0' '1' '2' '3' '4' '5' '6' '7' '8' '9'}))
        if obj.state~=LabelState.LABEL
          iPt = str2double(key);
          if iPt==0
            iPt = 10;
          end
          iPt = iPt+obj.kpfIPtFor1Key-1;
          if iPt>obj.nPts || iPt <= obj.nold
            return;
          end
          %obj.clearSelected(iPt);
          obj.toggleSelectPoint(iPt);
        end
      else
        tfKPused = false;
      end
    end
    
    function h = getLabelingHelp(obj) 
      h = {};
      h{end+1} = 'Click new keypoints to add in order.';
      h{end+1} = '';
      h{end+1} = 'Navigate to a frame that is partially labeled.';
      h{end+1} = 'Click to add the new points in order.';
      h{end+1} = 'Adjust as you like, either with the keypoint numbers or click to select.';
      h{end+1} = 'Click the Next button to advance to the next partially labeled frame.';
      h{end+1} = '';
      h1 = getLabelingHelp@LabelCore(obj);
      h = [h(:);h1(:)];
    end
    
  end
  
  methods
    
    function newFrameTarget(obj,iFrm,iTgt)
      % React to new frame or target. If a frame is not labeled, then start 
      % fresh in Label state. Otherwise, start in Accepted state with saved 
      % labels.
      
      %ticinfo = tic;
      [tflabeled,lpos,lpostag] = obj.labeler.labelPosIsPtLabeled(iFrm,iTgt);
      obj.nPtsLabeled = nnz(tflabeled);
      
      obj.assignLabelCoords(lpos,'lblTags',lpostag);
      obj.iPtMove = nan;
      
      if obj.nPtsLabeled < obj.nold,
        obj.beginAccepted(false);
      elseif obj.nPtsLabeled == obj.nold,
        obj.beginAdd(lpos,false);
      else
        obj.beginAccepted(false);
      end
    end
    
    function beginAdd(obj,lpos,tfClearLabels)
      % Enter Label state and clear all mode1 label state for current
      % frame/target
      
      set(obj.tbAccept,'BackgroundColor',[0.0 0.0 0.4],...
        'String','Adding','Enable','off','Value',0);
      if ~isempty(lpos),
        obj.assignLabelCoords(lpos);
      end
      obj.nPtsLabeled = obj.nold;
      obj.iPtMove = nan;
      obj.tfOcc(obj.nold+1:end) = false;
      obj.tfEstOcc(obj.nold+1:end) = false;
      obj.tfSel(obj.nold+1:end) = false;
      set(obj.hPts(ishandle(obj.hPts)),'HitTest','off');
      if tfClearLabels
        obj.labeler.labelPosClearPoints(obj.nold+1:obj.nPts);
        [~,lpos,~] = obj.labeler.labelPosIsPtLabeled(obj.labeler.currFrame,obj.labeler.currTarget);
        obj.assignLabelCoords(lpos);
      end
      obj.state = LabelState.LABEL;      
    end
    
%     function adjust2Label(obj)
%       % enter LABEL from ADJUST
%       set(obj.tbAccept,'BackgroundColor',[0.4 0.0 0.0],...
%         'String','Unlabeled','Enable','off','Value',0);      
%       obj.iPtMove = nan;
%       obj.state = LabelState.LABEL;      
%     end      
%        
%     function beginAdjust(obj)
%       % Enter adjustment state for current frame/target
%       
%       assert(obj.nPtsLabeled==obj.nPts);
%       obj.iPtMove = nan;
%       set(obj.tbAccept,'BackgroundColor',[0.6,0,0],'String','Accept',...
%         'Value',0,'Enable','off');
%       obj.state = LabelState.ADJUST;
%     end
    
    function beginAccepted(obj,tfSetLabelPos)
      % Enter accepted state (for current frame)
      
      if tfSetLabelPos
        xy = obj.getLabelCoords();
        obj.labeler.labelPosSet(xy);
        obj.setLabelPosTagFromEstOcc();
      end
      % KB 20181029: moved this here from beginAdjust as I remove adjust
      % mode
      hptsadd = obj.hPts(obj.nold+1:end);
      set(hptsadd(ishandle(hptsadd)),'HitTest','on');
      obj.iPtMove = nan;
      obj.clearSelected();
      set(obj.tbAccept,'BackgroundColor',[0,0.4,0],'String','Next',...
        'Value',1,'Enable','on');
      obj.state = LabelState.ACCEPTED;
    end
    
    function storeLabels(obj)
      
      xy = obj.getLabelCoords();
      obj.labeler.labelPosSet(xy);
      obj.setLabelPosTagFromEstOcc();
      
    end
    
    % C+P
%     function refreshTxLabelCoreAux(obj)
%       iPt0 = obj.kpfIPtFor1Key;
%       iPt1 = iPt0+9;
%       str = sprintf('Hotkeys 1-9,0 map to points %d-%d, ` (backquote) toggles',iPt0,iPt1);
%       obj.txLblCoreAux.String = str;      
%     end
    
    function toggleEstOccPoint(obj,iPt)
      obj.tfEstOcc(iPt) = ~obj.tfEstOcc(iPt);
      assert(~(obj.tfEstOcc(iPt) && obj.tfOcc(iPt)));
      obj.refreshPtMarkers('iPts',iPt);
      if obj.state==LabelState.ACCEPTED
        % KB 20181029: removing adjust state
        %obj.beginAdjust();
      end
    end

  end
  
end
