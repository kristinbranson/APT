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
        
    maxNumTgts = 5
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
    %kpfIPtFor1Key; % scalar positive integer. This is the point index that
                   % the '1' hotkey maps to, eg typically this will take the 
                   % values 1, 11, 21, ...
    
    unsupportedKPFFns = {'tbAccept'}; % cell array of field names for objects that have general keypressfcn 
                                      % callbacks but are not supported for this LabelCore
                   
  end
  properties (SetObservable)
    % two-click align
    % alt to using <shift>-a and <shift>-d for camroll
    tcOn = false; % scalar logical, true => two-click is on
  end
  properties
    % remainder applies when tcOn==true
    tcipt = 0; % 0, 1, or 2 depending on current number of two-click pts clicked
    tcHpts % [1] line handle for tc pts
    tcHptsPV = struct('Color','r','marker','+','markersize',10,'linewidth',2);
    tcShow = false; % scalar logical. true => leave tc points showing during lbl    
    tc_prev_axis = []; % to reset to prev view once the 2 click labeling is over
  end
  
  methods
    
    function obj = LabelCoreSeqMA(varargin)
      obj = obj@LabelCore(varargin{:});

      obj.roiAddButtons();
      obj.addMAbuttons();
      obj.tv = TrackingVisualizerMT(obj.labeler,'labelPointsPlotInfo',...
        'lblCoreSeqMA');
      obj.tv.doPch = true;
      obj.tv.vizInit('ntgts',obj.maxNumTgts);

      obj.roiInit();

      obj.labeler.currImHud.updateReadoutFields('hasTgt',true);
      % obj.labeler.gdata.axes_curr.Toolbar.Visible = 1;  
        % Commented out line above b/c doesn't respect menu setting.  
        % -- ALT, 2025-06-02
      obj.tcOn = obj.labeler.isTwoClickAlign;
    end
    function addMAbuttons(obj)
      btn = obj.pbRoiNew;
      YOFF_NORMALIZED = .01;
      pos = btn.Position;
      pos(2) = pos(2) + pos(4) + YOFF_NORMALIZED;
      
      pb = uicontrol(...
        'parent',obj.hFig(1),...
        'style','pushbutton',...
        'units',btn.Units,...
        'position',pos,...
        'fontunits',btn.FontUnits,...
        'fontsize',btn.FontSize,...
        'fontweight',btn.FontWeight,...
        'backgroundcolor',obj.CLR_NEW_TGT,...
        'string','New Target',...
        'callback',@(s,e)obj.cbkNewTgt() ...      
      );
      obj.pbNewTgt = pb;
      
      btn = obj.pbRoiEdit;
      pos = btn.Position;
      pos(2) = pos(2) + pos(4) + YOFF_NORMALIZED;
      pb = uicontrol(...
        'parent',obj.hFig(1),...
        'style','pushbutton',...
        'units',btn.Units,...
        'position',pos,...
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
      delete(obj.tv);
      delete(obj.pbNewTgt);
      delete(obj.pbDelTgt);
      delete(obj.pbRoiEdit);
      delete(obj.pbRoiNew);
      delete(obj.roiRectDrawer);
      deleteValidGraphicsHandles(obj.tcHpts);
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
      obj.enableControls();
      
      % MK 20220902, not sure where is the best place to add extra points
      % when projAddLandmarks is used.
      % Realized that add landmarks isn't enabled for multi-animal yet. 
      % when we do this might be handy, so keeping it around
%       if obj.nPts ~= obj.tv.nPts
%         delete(obj.tv);
%         obj.tv = TrackingVisualizerMT(obj.labeler,'labelPointsPlotInfo',...
%           'lblCoreSeqMA');
%         obj.tv.doPch = true;
%         obj.tv.vizInit('ntgts',obj.maxNumTgts);
%       end
    end
    
    function newFrame(obj,iFrm0,iFrm1,iTgt,tfForceUpdate) %#ok<INUSL>
      if nargin < 5
        tfForceUpdate = false;
      end
      obj.newFrameTarget(iFrm1,iTgt,tfForceUpdate);
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm) %#ok<INUSL>
      obj.newFrameTarget(iFrm,iTgt1);
    end
    
    function newFrameAndTarget(obj,~,iFrm1,~,iTgt1,tfForceUpdate)
      if nargin < 6
        tfForceUpdate = false;
      end
      obj.newFrameTarget(iFrm1,iTgt1,tfForceUpdate);
    end
    
    function clearLabels(obj)
      assert(false,'Nonproduction codepath');
    end
        
    function unAcceptLabels(obj)
      assert(false,'Nonproduction codepath');
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
            lObj = obj.labeler;
            pos = get(obj.hAx,'CurrentPoint');
            pos = pos(1,1:2);
            obj.assignLabelCoordsIRaw(pos,iSel);
            lObj.labelPosSetI(pos,iSel);
            obj.tfEstOcc(iSel) = tfShift; % following toggleSelectPoint call will call refreshPtMarkers
            obj.toggleSelectPoint(iSel);
            if obj.tfOcc(iSel)
              obj.tfOcc(iSel) = false;
              obj.refreshOccludedPts();
            end
            % estOcc status unchanged            
            
            % this is necessary to redraw any patches as appropriate
            [xy,tfeo] = obj.getLabelCoords(nan); % use nan for fully-occed so ROIs are drawn correctly
            iTgt = lObj.currTarget;
            obj.tv.updateTrackResI(xy,tfeo,iTgt);
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
            lObj = obj.labeler;
            obj.assignLabelCoordsIRaw([nan nan],iSel); % refreshOccludedPts() also sets pt coords, but this also updates skel
            lObj.labelPosSetIFullyOcc(iSel);
            if obj.tfEstOcc(iSel)
              obj.tfEstOcc(iSel) = false; 
              % following toggleSelectPoint call will call refreshPtMarkers
            end
            obj.toggleSelectPoint(iSel);
            obj.tfOcc(iSel) = true;
            obj.refreshOccludedPts();            
            % estOcc status unchanged
            
            % this is necessary to redraw any patches as appropriate
            [xy,tfeo] = obj.getLabelCoords(nan); % use nan for fully-occed so ROIs are drawn correctly
            iTgt = lObj.currTarget;
            obj.tv.updateTrackResI(xy,tfeo,iTgt);
          end
      end
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
          obj.tc_prev_axis = obj.controller.videoCurrentAxis() ;
          lObj.controller_.videoCenterOnCurrTarget(xc,yc,th)
          rad = 2*sqrt(dx.^2+dy.^2);
          lObj.controller_.videoZoom(rad);
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
        
    function ptBDF(obj,src,evt)
      %disp('ptbdf');
      if ~obj.labeler.isReady || evt.Button>1
        return;
      end
      if obj.isPanZoom(),
        return;
      end

      tf = obj.anyPointSelected();
      obj.labeler.unsetdrag();
      iPt = get(src,'UserData');
      obj.toggleSelectPoint(iPt);
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
      
      if ismember(gco,obj.labeler.controller_.tvTrx_.hTrx)
        return;
      end
      if obj.state == LabelState.ACCEPTED && ~isempty(obj.iPtMove) && ...
          ~isnan(obj.iPtMove)
        obj.toggleSelectPoint(obj.iPtMove);
        obj.iPtMove = nan;
        obj.storeLabels();
        [xy,tfeo] = obj.getLabelCoords();
        iTgt = obj.labeler.currTarget;
        obj.tv.updateTrackResI(xy,tfeo,iTgt);

      end
    end
    
    function shortcuts = LabelShortcuts(obj)

      shortcuts = cell(0,3);

      shortcuts{end+1,1} = 'New target';
      shortcuts{end,2} = 'w';
      shortcuts{end,3} = {'Ctrl'};

      shortcuts{end+1,1} = 'Undo last label click';
      shortcuts{end,2} = 'z';
      shortcuts{end,3} = {'Ctrl'};

      shortcuts{end+1,1} = 'Toggle whether selected kpt is occluded';
      shortcuts{end,2} = 'o';
      shortcuts{end,3} = {};
      
      shortcuts{end+1,1} = 'Rotate axes CCW by 2 degrees';
      shortcuts{end,2} = 'A';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = 'Rotate axes CW by 2 degrees';
      shortcuts{end,2} = 'D';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = 'Toggle whether panning tool is on';
      shortcuts{end,2} = 'a';
      shortcuts{end,3} = {'Ctrl'};

      shortcuts{end+1,1} = 'Toggle whether panning tool is on';
      shortcuts{end,2} = 'd';
      shortcuts{end,3} = {'Ctrl'};

      shortcuts{end+1,1} = 'Forward one frame';
      shortcuts{end,2} = '= or d';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = 'Backward one frame';
      shortcuts{end,2} = '- or a';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = 'Un/Select kpt of current target';
      shortcuts{end,2} = '0-9';
      shortcuts{end,3} = {};
      
      shortcuts{end+1,1} = 'Toggle which kpts 0-9 correspond to';
      shortcuts{end,2} = '`';
      shortcuts{end,3} = {};

      rightpx = obj.controller.videoCurrentRightVec;
      rightpx = rightpx(1);
      uppx = obj.controller.videoCurrentUpVec;
      uppx = abs(uppx(2));

      shortcuts{end+1,1} = sprintf('If kpt selected, move right by %.1f px, ow forward one frame',rightpx);
      shortcuts{end,2} = 'Right arrow';
      shortcuts{end,3} = {};
      
      shortcuts{end+1,1} = sprintf('If kpt selected, move left by %.1f px, ow back one frame',rightpx);
      shortcuts{end,2} = 'Left arrow';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = sprintf('If kpt selected, move up by %.1f px',uppx);
      shortcuts{end,2} = 'Up arrow';
      shortcuts{end,3} = {};
      
      shortcuts{end+1,1} = sprintf('If kpt selected, move down by %.1f px',uppx);
      shortcuts{end,2} = 'Down arrow';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = sprintf('If kpt selected, move right by %.1f px',10*rightpx);
      shortcuts{end,2} = '+';
      shortcuts{end,3} = {'Shift'};

      shortcuts{end+1,1} = sprintf('If kpt selected, move left by %.1f px',10*rightpx);
      shortcuts{end,2} = '-';
      shortcuts{end,3} = {'Shift'};

      shortcuts{end+1,1} = sprintf('If kpt selected, move left by %.1f px, ow go to next %s',10*rightpx,...
        obj.labeler.movieShiftArrowNavMode.prettyStr);
      shortcuts{end,2} = 'Left arrow';
      shortcuts{end,3} = {'Shift'};

      shortcuts{end+1,1} = sprintf('If kpt selected, move right by %.1f px, ow go to previous %s',10*rightpx,...
        obj.labeler.movieShiftArrowNavMode.prettyStr);
      shortcuts{end,2} = 'Right arrow';
      shortcuts{end,3} = {'Shift'};

      shortcuts{end+1,1} = sprintf('If kpt selected, move up by %.1f px',10*uppx);
      shortcuts{end,2} = 'Up arrow';
      shortcuts{end,3} = {'Shift'};

      shortcuts{end+1,1} = sprintf('If kpt selected, move down by %.1f px',10*uppx);
      shortcuts{end,2} = 'Down arrow';
      shortcuts{end,3} = {'Shift'};

      shortcuts{end+1,1} = 'Zoom in/out';
      shortcuts{end,2} = 'Mouse scroll';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = 'Pan view';
      shortcuts{end,2} = 'Mouse right-click-drag';
      shortcuts{end,3} = {};

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
      if tfShft && strcmp(key,'a'),
        camroll(obj.hAx,2);
      elseif tfShft && strcmp(key,'d'),
        camroll(obj.hAx,-2);
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
        lObj.frameUpGUI(tfCtrl);
      elseif any(strcmp(key,{'a' 'hyphen'})) && ~tfCtrl
        lObj.frameDownGUI(tfCtrl);
      elseif ~tfCtrl && any(strcmp(key,{'leftarrow' 'rightarrow' 'uparrow' 'downarrow'}))
        [tfSel,iSel] = obj.anyPointSelected();
        if tfSel % && ~obj.tfOcc(iSel)
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
          if tfShft
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
    function skeletonCosmeticsUpdated(obj)
      skeletonCosmeticsUpdated@LabelCore(obj);
      obj.tv.skeletonCosmeticsUpdated();
    end
    
    function preProcParamsChanged(obj)
      % react to preproc param mutation on lObj
      obj.tv.updatePches();
    end
  end
  
  methods % roi
    function roiInit(obj)
      obj.roiRectDrawer = RectDrawer(obj.hAx);
      %obj.roiAddButtons();
      obj.roiSetShow(false);
    end
    function roiAddButtons(obj)
      btn = obj.tbAccept;      
      pb = uicontrol(...
        'parent',obj.hFig(1),...
        'style','pushbutton',...
        'units',btn.Units,...
        'position',btn.Position,...
        'fontunits',btn.FontUnits,...
        'fontsize',btn.FontSize,...
        'fontweight',btn.FontWeight,...
        'backgroundcolor',obj.CLR_PBROINEW,...
        'string','New Label Box',...
        'units',btn.Units,...
        'callback',@(s,e)obj.cbkRoiNew() ...      
      );
      obj.pbRoiNew = pb;
      
      btn = obj.pbClear;
      pb = uicontrol(...
        'parent',obj.hFig(1),...
        'style','togglebutton',...
        'units',btn.Units,...
        'position',btn.Position,...
        'fontunits',btn.FontUnits,...
        'fontsize',btn.FontSize,...
        'fontweight',btn.FontWeight,...
        'backgroundcolor',obj.CLR_PBROIEDIT,...
        'string','Edit Label Boxes',...
        'units',btn.Units,...
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
      if tf
        % see newFrameTarget()
        lObj = obj.labeler;
        if ~lObj.isinit && lObj.hasMovie
          frm = lObj.currFrame;
          vroi = lObj.labelroiGet(frm);
          obj.roiRectDrawer.setRois(vroi);
          obj.roiUpdatePBEdit(false);
        end
      end
    end
    function cbkRoiNew(obj)
      assert(obj.roiShow);
      % obj.labeler.setStatus('Click and drag to add a box of pixels considered labeled. Hit Esc to cancel',false);
      set(obj.pbNewTgt,'Enable','off');
      set(obj.pbDelTgt,'Enable','off');
      set(obj.pbRoiNew,'Enable','off');
      set(obj.pbRoiEdit,'Enable','off');
      obj.roiRectDrawer.newRoiDraw();
      v = obj.roiRectDrawer.getRoisVerts();
      obj.labeler.labelroiSet(v);
      % obj.labeler.clearStatus();
      set(obj.pbNewTgt,'Enable','on');
      set(obj.pbDelTgt,'Enable','on');
      set(obj.pbRoiNew,'Enable','on');
      set(obj.pbRoiEdit,'Enable','on');
%       obj.roiUpdatePBEdit(true);
    end
    function cbkRoiEdit(obj)
      tfEditingNew = obj.pbRoiEdit.Value;
      % if tfEditingNew,
      %   obj.labeler.setStatus('Drag corners to move label boxes. Right click and select "Delete Rectangle" to delete.',false);
      % else
      %   obj.labeler.clearStatus();
      % end
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
        str = 'Edit Label Boxes';
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
      
      if obj.state == LabelState.LABEL,
        % cancel
        obj.beginAcceptedReset();
      else % ACCEPTED
        % add a new label
        ntgts = lObj.labelNumLabeledTgts();
        lObj.setTargetMA(ntgts+1);
        obj.newPrimaryTarget();
        lObj.updateTrxTable();
        obj.beginLabel();
      end
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
    
    function newFrameTarget(obj,iFrm,iTgt,tfForceUpdate)

      % React to new frame or target which might be labeled or unlabeled.
      %
      % PostCond: Accepted/Browse state

      % handle other targets
      [xy,occ] = obj.labeler.labelMAGetLabelsFrm(iFrm);
      xy(isinf(xy)) = nan; % inf => fully occluded. we replace with nan here so that ma ROIs are calculated correctly
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
          % KB 20210624 - removed this warning, not sure what it was on about. this always happens when you add a new video in multianimal mode
          %fprintf(2,'Setting lObj.currTarget to 0\n');
          lObj.setTargetMA(0);
        end
        obj.beginAcceptedReset();
      end
      obj.newPrimaryTarget();
      %fprintf('LabelCoreSeq.newFrameTarget 4: %f\n',toc(ticinfo)); 
      
      if obj.roiShow && ~lObj.gtIsGTMode
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

    function labelsHide(obj)
      obj.tv.setHideViz(true);
      labelsHide@LabelCore(obj);
    end
    
    function labelsShow(obj)
      obj.tv.setHideViz(false);
      labelsShow@LabelCore(obj);
    end

    function enableControls(obj)
      
      if obj.state == LabelState.LABEL,
        set(obj.pbNewTgt,'Enable','on');
        set(obj.pbDelTgt,'Enable','off');
        set(obj.pbRoiNew,'Enable','off');
        set(obj.pbRoiEdit,'Enable','off');
        set(obj.pbNewTgt,'String','Cancel');

      else
        set(obj.pbNewTgt,'Enable','on');
        set(obj.pbDelTgt,'Enable','on');
        set(obj.pbRoiNew,'Enable','on');
        set(obj.pbRoiEdit,'Enable','on');
        set(obj.pbNewTgt,'String','New Target');

      end
      
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
      %fprintf(1,'accept\n');
      lObj = obj.labeler;
%       ntgts = lObj.labelNumLabeledTgts();
%       lObj.setTargetMA(ntgts+1);
      obj.storeLabels();
      lObj.updateTrxTable();
      lObj.InitializePrevAxesTemplate();

      [xy,tfeo] = obj.getLabelCoords(nan); % use nan for fully-occed so ROIs are drawn correctly
      iTgt = lObj.currTarget;
      obj.tv.updateTrackResI(xy,tfeo,iTgt);
      % tv.hideTarget should already be set to lObj.currTarget
      obj.tv.hittest_on_all()
      if ~isempty(obj.labeler.tracker.trkVizer) && ...
          ~isempty(obj.labeler.tracker.trkVizer.tvmt)
        obj.labeler.tracker.trkVizer.tvmt.hittest_on_all();
      end
      if ~isempty(obj.labeler.tracker.trkVizer) && ...
          ~isempty(obj.labeler.tracker.trkVizer.tvtrx)
        obj.labeler.tracker.trkVizer.tvtrx.hittest_on_all();
      end
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
      if obj.tcOn && ~isempty(obj.tc_prev_axis)
        lObj.controller_.videoSetAxis(obj.tc_prev_axis);
          obj.tc_prev_axis = [];
      end
      obj.state = LabelState.ACCEPTED;
      obj.enableControls();
    end    
    function beginAcceptedReset(obj)
      % like beginAccepted, but reset first
      obj.resetState();
      obj.tcInit();
      lObj = obj.labeler;
      lObj.currImHud.hTxtTgt.BackgroundColor = [0 0 0];
      obj.state = LabelState.ACCEPTED;
      obj.enableControls();

    end
    function beginLabel(obj)
      % Enter Label state and clear all mode1 label state for current
      % frame/target   
      
      obj.resetState();
      lObj = obj.labeler;
      lObj.currImHud.hTxtTgt.BackgroundColor = obj.CLR_NEW_TGT;
      obj.state = LabelState.LABEL; 
      obj.tv.hittest_off_all()
      if ~isempty(obj.labeler.tracker.trkVizer) && ...
          ~isempty(obj.labeler.tracker.trkVizer.tvmt)
        obj.labeler.tracker.trkVizer.tvmt.hittest_off_all();
      end
      if ~isempty(obj.labeler.tracker.trkVizer) && ...
          ~isempty(obj.labeler.tracker.trkVizer.tvtrx)
        obj.labeler.tracker.trkVizer.tvtrx.hittest_off_all();
      end
      obj.enableControls();
    end
            
    function storeLabels(obj)
      %fprintf(1,'store\n');
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
    
% KB 20210907 - moved to LabelCore parent class
%     function refreshTxLabelCoreAux(obj)
%       iPt0 = obj.kpfIPtFor1Key;
%       iPt1 = iPt0+9;
%       str = sprintf('Hotkeys 0-9 map to points %d-%d',iPt0,iPt1);
%       obj.txLblCoreAux.String = str;      
%     end
    
    function toggleEstOccPoint(obj,iPt)
      obj.tfEstOcc(iPt) = ~obj.tfEstOcc(iPt);
      assert(~(obj.tfEstOcc(iPt) && obj.tfOcc(iPt)));
      obj.refreshPtMarkers('iPts',iPt);
    end
    
    function h = getLabelingHelp(obj) 
      h = { ...
        'To{\bf add a target}: '
        ' - Push the New Target button.'
        };
      if obj.tcOn,
        h{end+1} = ' - Click two points on the new target to zoom in on it.';
        h{end+1} = '   Often, these points correspond to the animal''s head and tail.';
      end
      h{end+1} = ' - Click the locations of your keypoints in order.';
      h{end+1} = ' - Hold shift while clicking to annotate that a keypoint is occluded.';
      h{end+1} = ' - You do not need to label all animals in each frame. ';
      h{end+1} = '   the black boxes show regions of the image around your labeled';
      h{end+1} = '   animals. APT only uses these boxes for training. If another';
      h{end+1} = '   animal is inside one of your label boxes, you should label it.';
      h{end+1} = '';
      h{end+1} = 'Use{\bf Label Boxes} to specify image regions that are completely labeled. ';
      h{end+1} = '  This is important for teaching the classifier what a negative label is. ';
      h{end+1} = '  An image region is completely labeled if no keypoints in that region';
      h{end+1} = '  are unlabeled. You e.g. can draw a label box around parts of the image';
      h{end+1} = '  that do not contain animals to add negative training examples.';
      h{end+1} = ' - Click New Label Box to add a new label box.';
      h{end+1} = '';
      h{end+1} = 'To{\bf set zoom}, at any time, mouse-scroll to zoom and';
      h{end+1} = '  right-click-drag to pan.';
      h{end+1} = '  Type Ctrl + f to zoom out and show the full frame.';
      h{end+1} = '';
      h{end+1} = 'To{\bf adjust labeled keypoints}:';
      h{end+1} = ' - Select the corresponding target number from the "Targets" box. ';
      h{end+1} = ' - Click the point or type its number to select a point. ';
      h{end+1} = '   Once selected, click the new location or use the arrow keys';
      h{end+1} = '   to move it. ';
      h{end+1} = ' - Alternatively, you can click and drag the keypoint.';
      h{end+1} = '';
      h{end+1} = 'To{\bf edit Label Boxes}:';
      h{end+1} = ' - Click Edit Label Boxes to enable editing. ';
      h{end+1} = ' - Drag the corners of a box to move or resize it.';
      h{end+1} = ' - Right-click the box and select Remove Rectangle to delete it.';
      h{end+1} = ' - Re-click Edit Label Boxes to register your changes.';
      h{end+1} = '';      

      h1 = getLabelingHelp@LabelCore(obj);
      h = [h(:);h1(:)];

    end

  end
  
end
