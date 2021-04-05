classdef LabelCoreSeqMA < LabelCore
% MA-Seq labeling  
  
  % Labeling states: Accepted/Browse and Label.
  %
  % During Browse, existing targets/labels (if any) are shown. Selection of 
  % a primary/focus target can occur eg via targets table. When this 
  % occurs, non-primary targets are grayed. Landmarks in the primary tgt
  % can be adjusted either by click-drag or by selectin/hotkey-ptBDF/arrows.
  %
  % During Browse, there may be no primary tgt. In that case no adjustments 
  % to existing landmarks can be made. 
  %
  % The primary target is always the same as lObj.currTarget.
  %
  % The Label state as in Seq mode. To enter the Label state, for now one
  % must explictly click the New Target button. Once a target is labeled,
  % labels are written to lObj and Browse is entered with the just-labeled 
  % target set as primary.
  %
  % pbDelTarget can be done either i) during Browse when a primary
  % target is set, in which case that target is removed (and no target is
  % set to primary); or iied) during Label, in which case the current target
  % is canceled and Browse-with-no-primary is entered.
  %
  % Changing targets or frames in the middle of Label is equivalent to
  % hitting pbRemoveTarget first as no new labels are written to lObj.
  
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
    tv % scalar TrackingVisualizerMT
    CLR_NEW_TGT = [0.470588235294118 0.670588235294118 0.188235294117647];
    CLR_DEL_TGT = [0.929411764705882 0.690196078431373 0.129411764705882];
    
    pbRoiNew
    pbRoiEdit
    roiShow = false; % show pbRoi* or not
    roiRectDrawer
    CLR_PBROINEW = [0.6941 .5082 .7365]; % [0.4902 0.1804 0.5608] see cropmode buttons
    CLR_PBROIEDIT = [0.4000 0.6706 0.8447]; % etc [0 0.4510 0.7412];
  end
  
  properties
    iPtMove; % scalar. Either nan, or index of pt being moved
    nPtsLabeled; % scalar integer. 0..nPts, or inf.

    % Templatemode-like behavior in 'adjust' and 'accepted' stages
    kpfIPtFor1Key; % scalar positive integer. This is the point index that
                   % the '1' hotkey maps to, eg typically this will take the 
                   % values 1, 11, 21, ...
                   
    % two-click align
    % alt to using <shift>-a and <shift>-d for camroll
    tcOn = false; % scalar logical, true => two-click is on
    % remainder applies when tcOn==true
    tcipt = 0; % 0, 1, or 2 depending on current number of two-click pts clicked
    tcHpts % [1] line handle for tc pts
    tcHptsPV = struct('Color','r','marker','+','markersize',10,'linewidth',2);
    tcShow = false; % scalar logical. true => leave tc points showing during lbl    
  end
  
  methods
    function set.kpfIPtFor1Key(obj,val)
      obj.kpfIPtFor1Key = val;
      obj.refreshTxLabelCoreAux();
    end
  end
  
  methods
    
    function obj = LabelCoreSeqMA(varargin)
      obj = obj@LabelCore(varargin{:});

      obj.addMAbuttons();
      obj.tv = TrackingVisualizerMT(obj.labeler,'lblCoreSeqMA');
      obj.tv.doPch = true;
      obj.tv.vizInit('ntgts',obj.maxNumTgts);

      obj.roiInit();

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
      pb = uicontrol(...
        'parent',obj.hFig(1),...
        'style','pushbutton',...
        'units',btn.Units,...
        'position',btn.Position,...
        'fontunits',btn.FontUnits,...
        'fontsize',btn.FontSize,...
        'fontweight',btn.FontWeight,...
        'backgroundcolor',obj.CLR_DEL_TGT,...
        'string','Remove Target',...
        'callback',@(s,e)obj.cbkDelTgt() ... 
      );
      obj.pbDelTgt = pb;
    end
    
    function delete(obj)
      delete(obj.pbNewTgt);
      delete(obj.pbDelTgt);
      delete(obj.pbRoiEdit);
      delete(obj.pbRoiNew);
      deleteValidHandles(obj.tcHpts);
    end
    
    function tcInit(obj)
      obj.tcipt = 0;
      if ~isempty(obj.tcHpts)
        set(obj.tcHpts,'XData',nan,'YData',nan);
      else
        obj.tcHpts = plot(obj.hAx,nan,nan);
        set(obj.tcHpts,obj.tcHptsPV);
      end
      % tcShow unchanged
    end
    function initHook(obj)
      obj.tcInit();
      
      obj.txLblCoreAux.Visible = 'on';
      obj.kpfIPtFor1Key = 1;
      obj.refreshTxLabelCoreAux();
      
      obj.state = LabelState.ACCEPTED; 
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
        
    function unAcceptLabels(obj)
      assert(false,'Nonproduction codepath');
    end
    
    function axBDF(obj,src,evt) %#ok<INUSD>
      if ~obj.labeler.isReady
        return;
      end
      
      mod = obj.hFig.CurrentModifier;
      tfShift = any(strcmp(mod,'shift'));
      switch obj.state
        case LabelState.LABEL
          if obj.tcOn && obj.tcipt<2
            pos = get(obj.hAx,'CurrentPoint');
            pos = pos(1,1:2);
            obj.hlpAxBDFTwoClick(pos);
            return
          end
          obj.hlpAxBDFLabelState(false,tfShift);          
        case LabelState.ACCEPTED
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
          
          end
      end
    end
    
    function axOccBDF(obj,~,~)
      assert(false,'Fully-occluded labels currently unsupported');
      
%       if ~obj.labeler.isReady,
%         return;
%       end
%       
%       mod = obj.hFig.CurrentModifier;
%       tfShift = any(strcmp(mod,'shift'));
% 
%       switch obj.state
%         case LabelState.LABEL
%           obj.hlpAxBDFLabelState(true,tfShift);
%         case {LabelState.ADJUST LabelState.ACCEPTED}
%           [tf,iSel] = obj.anyPointSelected();
%           if tf
%             if obj.tfEstOcc(iSel)
%               obj.tfEstOcc(iSel) = false; 
%               % following toggleSelectPoint call will call refreshPtMarkers
%             end
%             obj.toggleSelectPoint(iSel);        
%             obj.tfOcc(iSel) = true;
%             obj.refreshOccludedPts();
%             % estOcc status unchanged
%             if obj.state==LabelState.ACCEPTED
%               % KB 20181029: removing adjust state
%               %obj.beginAdjust();
%             end
%           end
%       end
    end
       
    function hlpAxBDFTwoClick(obj,xy)
      h = obj.tcHpts;
      switch obj.tcipt
        case 0
          set(h,'XData',xy(1),'YData',xy(2));
          obj.tcipt = 1;
        case 1
          x0 = h.XData;
          y0 = h.YData;
          set(h,'XData',[x0 xy(1)],'YData',[y0 xy(2)]);
          obj.tcipt = 2;
          
          xc = (x0+xy(1))/2;
          yc = (y0+xy(2))/2;
          dx = x0-xy(1);
          dy = y0-xy(2);
          th = atan2(dy,dx);
          lObj = obj.labeler;
          lObj.videoCenterOnCurrTarget(xc,yc,th)
          rad = 2*sqrt(dx.^2+dy.^2);
          lObj.videoZoom(rad);
          if ~obj.tcShow
            set(h,'XData',nan,'YData',nan);
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
        obj.acceptLabels();
      end
    end
    
    function undoLastLabel(obj)
      switch obj.state
        case LabelState.LABEL
          nlbled = obj.nPtsLabeled;
          if nlbled>0
            obj.tfSel(nlbled) = false;
            obj.tfEstOcc(nlbled) = false;
            obj.tfOcc(nlbled) = false;
            obj.refreshOccludedPts();
            obj.refreshPtMarkers('iPts',nlbled,'doPtsOcc',true);
            obj.assignLabelCoordsIRaw([nan nan],nlbled);
            obj.nPtsLabeled = nlbled-1;
          end          
      end
    end
        
    function ptBDF(obj,src,~)
      disp('ptbdf');
      if ~obj.labeler.isReady,
        return;
      end
      tf = obj.anyPointSelected();
      if tf
        % none
      else
        switch obj.state
          case LabelState.ACCEPTED
            iPt = get(src,'UserData');
            obj.iPtMove = iPt;
        end
      end
    end
    
    function wbmf(obj,~,~)
      if isempty(obj.state) || ~obj.labeler.isReady
        return;
      end
      if obj.state == LabelState.ACCEPTED
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
      if ~obj.labeler.isReady
        return;
      end
      
      if ismember(gco,obj.labeler.tvTrx.hTrx)
        return;
      end
      if obj.state == LabelState.ACCEPTED && ~isempty(obj.iPtMove) && ...
          ~isnan(obj.iPtMove)
        obj.iPtMove = nan;
        obj.storeLabels();
      end
    end
    
    function tfKPused = kpf(obj,~,evt)
      if ~obj.labeler.isReady
        return;
      end
      
      key = evt.Key;
      modifier = evt.Modifier;
      tfCtrl = ismember('control',modifier);
      tfShft = any(strcmp('shift',modifier));

      tfKPused = true;
      lObj = obj.labeler;
      if tfShft
        switch key
          case 'a'
            camroll(obj.hAx,2);
          case 'd'
            camroll(obj.hAx,-2);
        end
      elseif strcmp(key,'w') && tfCtrl
        obj.cbkNewTgt();
      elseif strcmp(key,'z') && tfCtrl
        obj.undoLastLabel();
      elseif strcmp(key,'o') && ~tfCtrl
        [tfSel,iSel] = obj.anyPointSelected();
        if tfSel
          obj.toggleEstOccPoint(iSel);
        end
        if obj.state == LabelState.ACCEPTED,
          obj.storeLabels();
        end
      elseif any(strcmp(key,{'d' 'equal'})) && ~tfCtrl
        lObj.frameUp(tfCtrl);
      elseif any(strcmp(key,{'a' 'hyphen'})) && ~tfCtrl
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
          
          % xxx no storeLabels()?
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
      h = { '' };
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
    
    function updateColors(obj,colors)
      updateColors@LabelCore(obj,colors);
      obj.tv.updateLandmarkColors(colors);
    end
    function updateMarkerCosmetics(obj,pvMarker)
      updateMarkerCosmetics@LabelCore(obj,pvMarker);
      obj.tv.setMarkerCosmetics(pvMarker);
    end
    function updateTextLabelCosmetics(obj,pvText,txtoffset)
      updateTextLabelCosmetics@LabelCore(obj,pvText,txtoffset);
      obj.tv.setTextCosmetics(pvText);
      obj.tv.setTextOffset(txtoffset);
    end
    
    function preProcParamsChanged(obj)
      % react to preproc param mutation on lObj
      obj.tv.updatePches();
    end
  end
  
  methods % roi
    function roiInit(obj)
      obj.roiRectDrawer = RectDrawer(obj.hAx);
      obj.roiAddButtons();
      obj.roiSetShow(false);
    end
    function roiAddButtons(obj)
      btn = obj.pbNewTgt;
      YOFF_NORMALIZED = .01;
      pos = btn.Position;
      pos(2) = pos(2) + pos(4) + YOFF_NORMALIZED;
      
      pb = uicontrol(...
        'parent',obj.hFig(1),...
        'style','pushbutton',...
        'units',btn.Units,...
        'position',btn.Position,...
        'fontunits',btn.FontUnits,...
        'fontsize',btn.FontSize,...
        'fontweight',btn.FontWeight,...
        'backgroundcolor',obj.CLR_PBROINEW,...
        'string','New ROI',...
        'units',btn.Units,...
        'position',pos,...
        'callback',@(s,e)obj.cbkRoiNew() ...      
      );
      obj.pbRoiNew = pb;
      
      btn = obj.pbClear;
      pos = btn.Position;
      pos(2) = pos(2) + pos(4) + YOFF_NORMALIZED;
      pb = uicontrol(...
        'parent',obj.hFig(1),...
        'style','togglebutton',...
        'units',btn.Units,...
        'position',btn.Position,...
        'fontunits',btn.FontUnits,...
        'fontsize',btn.FontSize,...
        'fontweight',btn.FontWeight,...
        'backgroundcolor',obj.CLR_PBROIEDIT,...
        'string','Edit ROIs',...
        'units',btn.Units,...
        'position',pos,...
        'callback',@(s,e)obj.cbkRoiEdit() ... 
      );
      obj.pbRoiEdit = pb;
    end
    function roiSetShow(obj,tf)
      onoff = onIff(tf);
      obj.pbRoiEdit.Visible = onoff;
      obj.pbRoiNew.Visible = onoff;
      obj.roiRectDrawer.setShowRois(tf);
      obj.roiShow = tf;
    end
    function cbkRoiNew(obj)
      assert(obj.roiShow);
      obj.roiRectDrawer.newRoiDraw();
      obj.roiUpdatePBEdit(true);
    end
    function cbkRoiEdit(obj)
      tfEditingNew = obj.pbRoiEdit.Value;
      rrd = obj.roiRectDrawer;
      rrd.setEdit(tfEditingNew);
      if ~tfEditingNew
        % write to db/Labeler
        v = rrd.getRoisVerts();
        obj.labeler.labelroiSet(v);
      end
      obj.roiUpdatePBEdit(tfEditingNew);
    end
    function roiUpdatePBEdit(obj,tf)
      if tf
        str = 'Done Editing'; 
        val = 1;
      else
        str = 'Edit ROIs';
        val = 0;
      end
      set(obj.pbRoiEdit,'Value',val,'String',str);
    end
  end
  
  methods % tc
    function setTwoClickOn(obj,tfon)
      if obj.state==LabelState.LABEL
        error('Please finish labeling the current animal.');
      end
      obj.tcInit();
      obj.tcOn = tfon;
    end
  end
  
  methods
    
    function cbkNewTgt(obj)
      lObj = obj.labeler;
      ntgts = lObj.labelNumLabeledTgts();
      lObj.setTargetMA(ntgts+1);
      obj.newPrimaryTarget();
      lObj.updateTrxTable();
      obj.beginLabel();
    end
    
    function cbkDelTgt(obj)
      lObj = obj.labeler;
      if obj.state==LabelState.ACCEPTED
        ntgts = lObj.labelPosClearWithCompact_New();
        iTgt = lObj.currTarget;
        if iTgt>ntgts
          lObj.setTargetMA(ntgts);          
        end
      end
      obj.newFrameTarget(lObj.currFrame,lObj.currTarget);
%       else % LABEL
%         iTgt = lObj.currTarget - 1;
%       end
%       lObj.setTargetMA(iTgt);
%       obj.newPrimaryTarget();
%       lObj.updateTrxTable();
%       obj.beginAcceptedReset();
    end
    
    function newFrameTarget(obj,iFrm,iTgt)
      % React to new frame or target which might be labeled or unlabeled.
      %
      % PostCond: Accepted/Browse state

      % handle other targets
      [xy,occ] = obj.labeler.labelMAGetLabelsFrm(iFrm);
      obj.tv.updateTrackRes(xy,occ);

      %ticinfo = tic;
      lObj = obj.labeler;
      [tflabeled,lpos,lpostag] = lObj.labelPosIsLabeled(iFrm,iTgt);
      if ~tflabeled
        % iTgt is not labeled, but we set the primary target to a labeled
        % frm if avail
        iTgts = lObj.labelPosIsLabeledFrm(iFrm);
        if ~isempty(iTgts)
          iTgt = min(iTgts); % TODO could take iTgt closest to existing iTgt
          [~,lpos,lpostag] = lObj.labelPosIsLabeled(iFrm,iTgt);
          tflabeled = true;
          lObj.setTargetMA(iTgt);
        end
      end
            
      %fprintf('LabelCoreSeq.newFrameTarget 1: %f\n',toc(ticinfo));ticinfo = tic;
      if tflabeled
        obj.nPtsLabeled = obj.nPts;
        obj.assignLabelCoords(lpos,'lblTags',lpostag);
        %fprintf('LabelCoreSeq.newFrameTarget 2: %f\n',toc(ticinfo));ticinfo = tic;
        obj.beginAccepted(); % Could possibly just call with true arg
        %fprintf('LabelCoreSeq.newFrameTarget 3: %f\n',toc(ticinfo));ticinfo = tic;
      else
        if iTgt~=0
          fprintf(2,'Setting lObj.currTarget to 0\n');
          lObj.setTargetMA(0);
        end
        obj.beginAcceptedReset();
      end
      obj.newPrimaryTarget();
      %fprintf('LabelCoreSeq.newFrameTarget 4: %f\n',toc(ticinfo)); 
      
      if obj.roiShow
        % Note, currently if roiShow is toggled, rois for the current
        % frame will not be shown until a frame-change.
        vroi = lObj.labelroiGet(iFrm);
        obj.roiRectDrawer.setRois(vroi);
        
        % changing frames always resets ROI editing state. Note that
        % changinging frames before clicking "Done Editing" will not save/
        % write ROI changes to Labeler.
        obj.roiUpdatePBEdit(false);
      end
    end
    
    function newPrimaryTarget(obj)
      % The 'primary target' for LabelCoreSeqMA always matches
      % lObj.currtarget.
      iTgt = obj.labeler.currTarget;
      if iTgt==0
        iTgt = []; % ie dont hide any targets
      end      
      obj.tv.updateHideTarget(iTgt); 
    end

    function resetState(obj)
      obj.assignLabelCoords(nan(obj.nPts,2));
      obj.nPtsLabeled = 0;
      obj.iPtMove = nan;
      obj.tfOcc(:) = false;
      obj.tfEstOcc(:) = false;
      obj.tfSel(:) = false;
    end
    
    function acceptLabels(obj)
      fprintf(1,'accept\n');
      lObj = obj.labeler;
%       ntgts = lObj.labelNumLabeledTgts();
%       lObj.setTargetMA(ntgts+1);
      obj.storeLabels();
      lObj.updateTrxTable();
      lObj.InitializePrevAxesTemplate();

      [xy,tfeo] = obj.getLabelCoords();
      iTgt = lObj.currTarget;
      obj.tv.updateTrackResI(xy,tfeo,iTgt);
      % tv.hideTarget should already be set to lObj.currTarget
      
      obj.beginAccepted();
    end

    function beginAccepted(obj) 
      % Enter accepted state. Preconds:
      % 1. Current primary labeling pts should already be set appropriately 
      % (eg via assignLabelCoords). If there is no current label, these
      % should be set to nan
      % 2. .tv.hideTarget should be set to current primary tgt (if avail),
      % or [] otherwise

      obj.iPtMove = nan;
      obj.clearSelected();
      obj.tcInit();
      lObj = obj.labeler;
      lObj.currImHud.hTxtTgt.BackgroundColor = [0 0 0];
      obj.state = LabelState.ACCEPTED;
    end    
    function beginAcceptedReset(obj)
      % like beginAccepted, but reset first
      obj.resetState();
      obj.tcInit();
      lObj = obj.labeler;
      lObj.currImHud.hTxtTgt.BackgroundColor = [0 0 0];
      obj.state = LabelState.ACCEPTED;
    end
    function beginLabel(obj)
      % Enter Label state and clear all mode1 label state for current
      % frame/target   
      
      obj.resetState();
      lObj = obj.labeler;
      lObj.currImHud.hTxtTgt.BackgroundColor = obj.CLR_NEW_TGT;
      obj.state = LabelState.LABEL;      
    end
            
    function storeLabels(obj)
      fprintf(1,'store\n');
      [xy,tfeo] = obj.getLabelCoords();
      obj.labeler.labelPosSet(xy,tfeo);
    end
    
%     function createNewTargetAndSetLabel(obj,xy,occ)
%       % Utility equivalent to
%       % 1. Pressing 'New Target'
%       % 2. Labeling all pts, possibly with occ, fully labeling tgt
%       %
%       % This util is for eg "set manual label to pred"
%       
%       obj.cbkNewTgt();
%       lObj = obj.labeler;
%       obj.storeLabels();
%       lObj.updateTrxTable();
%       lObj.InitializePrevAxesTemplate();
% 
%       [xy,tfeo] = obj.getLabelCoords();
%       iTgt = lObj.currTarget;
%       obj.tv.updateTrackResI(xy,tfeo,iTgt);
%       % tv.hideTarget should already be set to lObj.currTarget
%       
%       obj.beginAccepted();
%     end
    
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
    end

  end
  
end