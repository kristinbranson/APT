classdef LabelCoreSeqMA < LabelCore
% Sequential labeling  
  
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
    supportsSingleView = true;
    supportsMultiView = false;
    supportsCalibration = false;
    supportsMultiAnimal = true;
  end
        
  properties
    pbNewTgt % create a new target
    pbDelTgt % delete the current tgt
    
    maxNumTgts = 10
    tv % TrackingVisualizerMT
    CLR_NEW_TGT = [0.470588235294118 0.670588235294118 0.188235294117647];
  end
  properties
    
    iPtMove; % scalar. Either nan, or index of pt being moved
    nPtsLabeled; % scalar integer. 0..nPts, or inf.

    % Templatemode-like behavior in 'adjust' and 'accepted' stages
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
    
    %OK
    function obj = LabelCoreSeqMA(varargin)
      obj = obj@LabelCore(varargin{:});
      %set(obj.tbAccept,'Enable','off');
      obj.addMAbuttons();
      obj.tv = TrackingVisualizerMT(obj.labeler,'lblCoreSeqMA');
      obj.tv.doPch = true;
      obj.tv.vizInit('ntgts',obj.maxNumTgts);
      
      obj.labeler.currImHud.updateReadoutFields('hasTgt',true);
      obj.labeler.gdata.axes_curr.Toolbar.Visible = 1; 
    end
    function addMAbuttons(obj)
      btn = obj.tbAccept;
      pb = uicontrol(...
        'parent',obj.hFig(1),...
        'style','pushbutton',...
        'units',btn.Units,...
        'position',btn.Position,...
        'fontunits',btn.FontUnits,...
        'fontsize',btn.FontSize,...
        'fontweight',btn.FontWeight,...
        'backgroundcolor',obj.CLR_NEW_TGT,...
        'string','New Target',...
        'callback',@(s,e)obj.cbkNewTgt() ...      
      );
      obj.pbNewTgt = pb;
      
      btn = obj.pbClear;
      CLR = [0.929411764705882 0.690196078431373 0.129411764705882];
      pb = uicontrol(...
        'parent',obj.hFig(1),...
        'style','pushbutton',...
        'units',btn.Units,...
        'position',btn.Position,...
        'fontunits',btn.FontUnits,...
        'fontsize',btn.FontSize,...
        'fontweight',btn.FontWeight,...
        'backgroundcolor',CLR,...
        'string','Remove Target',...
        'callback',@(s,e)obj.cbkDelTgt() ... 
      );
      obj.pbDelTgt = pb;
    end
    
    function delete(obj)
      delete(obj.pbNewTgt);
      delete(obj.pbDelTgt);
    end
    
    function initHook(obj)
      obj.txLblCoreAux.Visible = 'on';
      obj.kpfIPtFor1Key = 1;
      obj.refreshTxLabelCoreAux();
      
      % AL 20190203 semi-hack. init to something/anything to avoid error 
      % with .state unset. Did the same with LabelCoreTemplate. A little
      % disturbing, something has changed with labelCore init or
      % Labeler.labelingInit but not clear what. Prob not a super-huge risk
      % low prob of significant data loss
      obj.state = LabelState.ADJUST; 
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
      assert(false,'Nonproduction codepath');
    end
    
    function acceptLabels(obj)
      fprintf(1,'accept\n');
      obj.storeLabels();
      lObj = obj.labeler;
      lObj.updateTrxTable();
      lObj.InitializePrevAxesTemplate();
      lObj.currImHud.hTxtTgt.BackgroundColor = [0 0 0];

%       xy = obj.getLabelCoords(); 
% 
%       ntgts = lObj.labelNumLabeledTgts(obj);
%       lObj.setTargetMA(ntgts+1);
%       lObj.labelPosSet(xy);
%       obj.setLabelPosTagFromEstOcc();  
    end
    
    function unAcceptLabels(obj)
      assert(false,'Nonproduction codepath');
    end
    
    %OKEND
    function axBDF(obj,src,evt) %#ok<INUSD>
      if ~obj.labeler.isReady,
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
    
    % XXX MA
    function axOccBDF(obj,~,~)
      assert(false,'Unsupported');
      
      if ~obj.labeler.isReady,
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
      % false in beginLabel();
      
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
        
    function ptBDF(obj,src,~)
      if ~obj.labeler.isReady,
        return;
      end
      tf = obj.anyPointSelected();
      if tf
        % none
      else
        switch obj.state
          case {LabelState.ADJUST LabelState.ACCEPTED}          
            iPt = get(src,'UserData');
            % KB 20181029: removing adjust state
%             if obj.state==LabelState.ACCEPTED
%               obj.beginAdjust();
%             end
            obj.iPtMove = iPt;
        end
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
      if ismember(gco,obj.labeler.hTrx),
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
              % KB 20181029: removing adjust state
              %obj.beginAdjust();
          end
        else
          tfKPused = false;
        end
      elseif strcmp(key,'backquote')
        iPt = obj.kpfIPtFor1Key+10;
        if iPt>obj.nPts
          iPt = 1;
        end
        obj.kpfIPtFor1Key = iPt;
      elseif any(strcmp(key,{'0' '1' '2' '3' '4' '5' '6' '7' '8' '9'}))
        if obj.state~=LabelState.LABEL
          iPt = str2double(key);
          if iPt==0
            iPt = 10;
          end
          iPt = iPt+obj.kpfIPtFor1Key-1;
          if iPt>obj.nPts
            return;
          end
          %obj.clearSelected(iPt);
          obj.toggleSelectPoint(iPt);
        end
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
    
    function updateSkeletonEdges(obj,varargin)
      updateSkeletonEdges@LabelCore(obj,varargin{:});
      se = obj.skeletonEdges;
      obj.tv.initAndUpdateSkeletonEdges(se);
    end
    function updateShowSkeleton(obj)
      updateShowSkeleton@LabelCore(obj);
      tf = obj.labeler.showSkeleton;
      obj.tv.setShowSkeleton(tf);
    end
  end
  
  methods
    
    function cbkNewTgt(obj)
      lObj = obj.labeler;
      ntgts = lObj.labelNumLabeledTgts();
      lObj.setTarget(ntgts+1);
      lObj.updateTrxTable();
      lObj.currImHud.hTxtTgt.BackgroundColor = obj.CLR_NEW_TGT;
    end
    
    function cbkDelTgt(obj)
      lObj = obj.labeler;
      %ntgts = lObj.labelNumLabeledTgts(obj);
      lObj.labelPosClearWithCompact_New(); % sets target
      lObj.updateTrxTable();
    end
    
    function newFrameTarget(obj,iFrm,iTgt)
      % React to new frame or target. If a frame is not labeled, then start 
      % fresh in Label state. Otherwise, start in Accepted state with saved 
      % labels.
      
      %ticinfo = tic;
      [tflabeled,lpos,lpostag] = obj.labeler.labelPosIsLabeled(iFrm,iTgt);
      %fprintf('LabelCoreSeq.newFrameTarget 1: %f\n',toc(ticinfo));ticinfo = tic;
      if tflabeled
        obj.nPtsLabeled = obj.nPts;
        obj.assignLabelCoords(lpos,'lblTags',lpostag);
        %fprintf('LabelCoreSeq.newFrameTarget 2: %f\n',toc(ticinfo));ticinfo = tic;
        obj.iPtMove = nan;
        obj.beginAccepted(); % Could possibly just call with true arg
        %fprintf('LabelCoreSeq.newFrameTarget 3: %f\n',toc(ticinfo));ticinfo = tic;
      else
        obj.beginLabel();
      end
      %fprintf('LabelCoreSeq.newFrameTarget 4: %f\n',toc(ticinfo));
      
      [xy,occ] = obj.labeler.labelMAGetLabelsFrm(iFrm,obj.maxNumTgts);
      obj.tv.updateTrackRes(xy,iTgt);
    end
    
    function beginLabel(obj)
      % Enter Label state and clear all mode1 label state for current
      % frame/target
      
%       set(obj.tbAccept,'BackgroundColor',[0.4 0.0 0.0],...
%         'String','Unlabeled','Enable','off','Value',0);
      obj.assignLabelCoords(nan(obj.nPts,2));
      obj.nPtsLabeled = 0;
      obj.iPtMove = nan;
      obj.tfOcc(:) = false;
      obj.tfEstOcc(:) = false;
      obj.tfSel(:) = false;
%       if tfClearLabels
%         obj.labeler.labelPosClear();
%       end
      obj.state = LabelState.LABEL;      
    end
    
%     function adjust2Label(obj)
%       % enter LABEL from ADJUST
%       set(obj.tbAccept,'BackgroundColor',[0.4 0.0 0.0],...
%         'String','Unlabeled','Enable','off','Value',0);      
%       obj.iPtMove = nan;
%       obj.state = LabelState.LABEL;      
%     end      
       
%     function beginAdjust(obj)
%       % Enter adjustment state for current frame/target
%       
%       assert(obj.nPtsLabeled==obj.nPts);
%       obj.iPtMove = nan;
%       set(obj.tbAccept,'BackgroundColor',[0.6,0,0],'String','Accept',...
%         'Value',0,'Enable','off');
%       obj.state = LabelState.ADJUST;
%     end
    
    function beginAccepted(obj) %,tfSetLabelPos)
      % Enter accepted state (for current frame)
%       
%       if tfSetLabelPos
%         xy = obj.getLabelCoords();
%         obj.labeler.labelPosSet(xy);
%         obj.setLabelPosTagFromEstOcc();
%       end
      % KB 20181029: moved this here from beginAdjust as I remove adjust
      % mode
      obj.iPtMove = nan;
      obj.clearSelected();
%       set(obj.tbAccept,'BackgroundColor',[0,0.4,0],'String','Labeled',...
%         'Value',1,'Enable','off');
      obj.state = LabelState.ACCEPTED;
    end
    
    function storeLabels(obj)
      fprintf(1,'store\n');
      xy = obj.getLabelCoords();
      obj.labeler.labelPosSet(xy);
      obj.setLabelPosTagFromEstOcc();      
    end
    
    % C+P
    function refreshTxLabelCoreAux(obj)
      iPt0 = obj.kpfIPtFor1Key;
      iPt1 = iPt0+9;
      str = sprintf('Hotkeys 0-9 map to points %d-%d',iPt0,iPt1);
      obj.txLblCoreAux.String = str;      
    end
    
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