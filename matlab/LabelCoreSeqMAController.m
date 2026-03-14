classdef LabelCoreSeqMAController < LabelCoreController
% Multi-animal sequential labeling controller
%
% Owns all graphics for multi-animal sequential labeling mode. Receives GUI
% callbacks, extracts GUI state (mouse position, modifiers), delegates data
% logic to LabelCoreSeqMAModel, and syncs graphics in response to model
% events.

  properties
    supportsSingleView = true ;
    supportsMultiView = false ;
    supportsCalibration = false ;
    supportsMultiAnimal = true ;
  end

  properties (Transient)
    pbNewTgt_               % pushbutton: create a new target
    pbDelTgt_               % pushbutton: delete the current tgt
    tv_                     % scalar TrackingVisualizerMT
    pbRoiNew_               % pushbutton: new label box
    pbRoiEdit_              % togglebutton: edit label boxes
    roiRectDrawer_          % scalar RectDrawer
    tcHpts_                 % [1] line handle for tc pts
  end

  properties (Constant)
    tcHptsPV_ = struct('Color', 'r', 'marker', '+', 'markersize', 10, 'linewidth', 2) ;
    CLR_PBROINEW_ = [0.6941 .5082 .7365] ;
    CLR_PBROIEDIT_ = [0.4000 0.6706 0.8447] ;
  end

  methods

    function obj = LabelCoreSeqMAController(labelerController, labeler, model)
      % Construct a LabelCoreSeqMAController.
      obj = obj@LabelCoreController(labelerController, labeler, model) ;
    end  % function

    function initHook(obj)
      % Initialize SeqMA-specific graphics: buttons, TrackingVisualizerMT,
      % ROI drawer, two-click handles, and model event listeners.

      mdl = obj.model_ ;
      lObj = obj.labeler_ ;

      obj.roiAddButtons() ;
      obj.addMAbuttons() ;

      tvm = TrackingVisualizerMTModel(lObj, 'labelPointsPlotInfo', 'lblCoreSeqMA') ;
      tvm.doPch = true ;
      obj.tv_ = TrackingVisualizerMT(obj.labelerController_, tvm) ;
      obj.tv_.vizInit('ntgts', mdl.maxNumTgts_) ;

      % Do this now that obj.tv_ is initialized
      se = obj.model_.skeletonEdges() ;
      obj.tv_.initAndUpdateSkeletonEdges(se) ;

      % And this
      tf = obj.labeler_.showSkeleton ;
      obj.tv_.setShowSkeleton(tf) ;
      
      obj.roiInit() ;

      lObj.currImHudModel.hasTgt = true ;
      lObj.notify('updateHudReadoutFields') ;
      mdl.tcOn_ = lObj.isTwoClickAlign ;

      obj.tcInitGraphics() ;

      obj.txLblCoreAux_.Visible = 'on' ;
      obj.refreshTxLabelCoreAux() ;

      obj.enableControls() ;

      % Register additional model listeners for SeqMA-specific events
      obj.listeners_ = [obj.listeners_ ; ...
        addlistener(mdl, 'updateTwoClickState',  @(s,e)obj.onUpdateTwoClickState()) ; ...
        addlistener(mdl, 'updateNewFrameTarget', @(s,e)obj.onUpdateNewFrameTarget()) ; ...
        addlistener(mdl, 'updateAccepted',       @(s,e)obj.onUpdateAccepted()) ; ...
        addlistener(mdl, 'updateAcceptedReset',   @(s,e)obj.onUpdateAcceptedReset()) ; ...
        addlistener(mdl, 'updateBeginLabel',     @(s,e)obj.onUpdateBeginLabel()) ; ...
        addlistener(mdl, 'restoreVideoAxis',   @(s,e)obj.onRestoreVideoAxis()) ; ...
      ] ;
    end  % function

    function delete(obj)
      % Clean up SeqMA-specific graphics handles.
      delete(obj.tv_) ;
      delete(obj.pbNewTgt_) ;
      delete(obj.pbDelTgt_) ;
      delete(obj.pbRoiEdit_) ;
      delete(obj.pbRoiNew_) ;
      delete(obj.roiRectDrawer_) ;
      deleteValidGraphicsHandles(obj.tcHpts_) ;
    end  % function

  end  % methods

  %% Model event handlers
  methods

    function onUpdateState(obj)
      % Sync button appearance and controls to model state.
      obj.enableControls() ;
    end  % function

    function onUpdateNewFrameTarget(obj)
      % Sync track visualization and ROI to new frame/target data.

      mdl = obj.model_ ;
      lObj = obj.labeler_ ;
      iFrm = lObj.currFrame ;

      % Update all MA track results
      [xy, occ] = lObj.labelMAGetLabelsFrm(iFrm) ;
      xy(isinf(xy)) = nan ; % inf => fully occluded; replace with nan so ROIs calculate correctly
      obj.tv_.updateTrackRes(xy, occ) ;

      % Update primary target highlight
      obj.newPrimaryTarget() ;

      % Update ROI display
      if lObj.showMaRoiAux && ~lObj.gtIsGTMode
        vroi = lObj.labelroiGet(iFrm) ;
        obj.roiRectDrawer_.setRois(vroi) ;
        obj.roiUpdatePBEdit(false) ;
      end
    end  % function

    function onUpdateAccepted(obj)
      % Respond to acceptLabels: update tv for current target, restore hittest.

      mdl = obj.model_ ;
      lObj = obj.labeler_ ;
      [xy, tfeo] = mdl.getLabelCoords(nan) ; % use nan for fully-occed so ROIs are drawn correctly
      iTgt = lObj.currTarget ;
      obj.tv_.updateTrackResI(xy, tfeo, iTgt) ;
      obj.tv_.hittest_on_all() ;
      tvPred = obj.labelerController_.tvTrkPred_ ;
      if ~isempty(tvPred)
        if isprop(tvPred, 'tvmt') && ~isempty(tvPred.tvmt)
          tvPred.tvmt.hittest_on_all() ;
        end
        if isprop(tvPred, 'tvtrx') && ~isempty(tvPred.tvtrx)
          tvPred.tvtrx.hittest_on_all() ;
        end
      end
    end  % function

    function onUpdateAcceptedReset(obj) %#ok<MANU>
      % Respond to beginAcceptedReset. Currently handled by onUpdateState.
    end  % function

    function onRestoreVideoAxis(obj)
      % Restore video axis after two-click align.
      mdl = obj.model_ ;
      obj.labelerController_.videoSetAxis(mdl.tc_prev_axis_) ;
    end  % function

    function onUpdateBeginLabel(obj)
      % Respond to entering Label state: update HUD and disable hittest.

      mdl = obj.model_ ;
      obj.labelerController_.currImHud.hTxtTgt.BackgroundColor = mdl.CLR_NEW_TGT ;
      obj.tv_.hittest_off_all() ;
      tvPred = obj.labelerController_.tvTrkPred_ ;
      if ~isempty(tvPred)
        if isprop(tvPred, 'tvmt') && ~isempty(tvPred.tvmt)
          tvPred.tvmt.hittest_off_all() ;
        end
        if isprop(tvPred, 'tvtrx') && ~isempty(tvPred.tvtrx)
          tvPred.tvtrx.hittest_off_all() ;
        end
      end
    end  % function

    function onUpdateLabelCoordsI(obj)
      % Sync single-point graphics and refresh markers for changed point.
      onUpdateLabelCoordsI@LabelCoreController(obj) ;
      iPt = obj.model_.lastChangedIPt ;
      obj.refreshPtMarkers('iPts', iPt) ;
    end  % function

  end  % methods

  %% GUI callback handlers
  methods

    function axBDF(obj, src, evt) %#ok<INUSL>
      % Handle axis button-down: place next point, two-click, or relocate.

      mdl = obj.model_ ;
      if ~obj.labeler_.isReady || evt.Button > 1
        return ;
      end
      if obj.isPanZoom()
        return ;
      end

      pos = evt.IntersectionPoint(1:2) ;
      mod = obj.hFig_(1).CurrentModifier ;
      tfShift = any(strcmp(mod, 'shift')) ;
      switch mdl.state
        case LabelState.LABEL
          if mdl.tcOn_ && mdl.tcipt_ < 2
            obj.hlpAxBDFTwoClick(pos) ;
            return ;
          end
          mdl.hlpAxBDFLabelState(false, tfShift, pos) ;
        case LabelState.ACCEPTED
          [tf, iSel] = mdl.anyPointSelected() ;
          if tf
            mdl.relocatePoint(iSel, pos, tfShift) ;
            % Update tv for this target
            [xy, tfeo] = mdl.getLabelCoords(nan) ;
            iTgt = obj.labeler_.currTarget ;
            obj.tv_.updateTrackResI(xy, tfeo, iTgt) ;
          end
        otherwise
          error('LabelCoreSeqMAController:unknownState', ...
                'Unknown state %s.', char(mdl.state)) ;
      end
    end  % function

    function axOccBDF(obj, ~, ~)
      % Handle occluded-axis button-down.

      mdl = obj.model_ ;
      if ~obj.labeler_.isReady
        return ;
      end
      if obj.isPanZoom()
        return ;
      end

      mod = obj.hFig_(1).CurrentModifier ;
      tfShift = any(strcmp(mod, 'shift')) ;

      switch mdl.state
        case LabelState.LABEL
          mdl.hlpAxBDFLabelState(true, tfShift, [nan nan]) ;
        case {LabelState.ADJUST, LabelState.ACCEPTED}
          [tf, iSel] = mdl.anyPointSelected() ;
          if tf
            mdl.occludeSelectedPoint(iSel) ;
            % Update tv for this target
            [xy, tfeo] = mdl.getLabelCoords(nan) ;
            iTgt = obj.labeler_.currTarget ;
            obj.tv_.updateTrackResI(xy, tfeo, iTgt) ;
          end
        otherwise
          error('LabelCoreSeqMAController:unknownState', ...
                'Unknown state %s.', char(mdl.state)) ;
      end
    end  % function

    function ptBDF(obj, src, evt)
      % Handle point button-down: select point and possibly start drag.

      mdl = obj.model_ ;
      if ~obj.labeler_.isReady || evt.Button > 1
        return ;
      end
      if obj.isPanZoom()
        return ;
      end

      tf = mdl.anyPointSelected() ;
      obj.labeler_.unsetdrag() ;
      iPt = get(src, 'UserData') ;
      mdl.toggleSelectPoint(iPt) ;
      if tf
        % none
      else
        switch mdl.state
          case LabelState.ACCEPTED
            mdl.iPtMove_ = iPt ;
          case LabelState.LABEL
            % No drag interaction during LABEL state
          otherwise
            error('LabelCoreSeqMAController:unknownState', ...
                  'Unknown state %s.', char(mdl.state)) ;
        end
      end
    end  % function

    function wbmf(obj, ~, ~)
      % Handle window button motion: drag selected point.
      % Bypasses event system for responsiveness during continuous drag.

      mdl = obj.model_ ;
      if isempty(mdl.state) || ~obj.labeler_.isReady
        return ;
      end
      if mdl.state == LabelState.ACCEPTED
        iPt = mdl.iPtMove_ ;
        if ~isnan(iPt)
          ax = obj.hAx_(1) ;
          tmp = get(ax, 'CurrentPoint') ;
          pos = tmp(1, 1:2) ;
          mdl.xy(iPt, :) = pos ;
          obj.syncPointGraphicsI(iPt) ;
        end
      end
    end  % function

    function wbuf(obj, ~, ~)
      % Handle window button up: end drag, persist labels.

      mdl = obj.model_ ;
      if ~obj.labeler_.isReady
        return ;
      end

      if ismember(gco, obj.labelerController_.tvTrx_.hTrx)
        return ;
      end
      if mdl.state == LabelState.ACCEPTED && ~isempty(mdl.iPtMove_) && ...
          ~isnan(mdl.iPtMove_)
        mdl.toggleSelectPoint(mdl.iPtMove_) ;
        mdl.iPtMove_ = nan ;
        mdl.storeLabels() ;
        [xy, tfeo] = mdl.getLabelCoords() ;
        iTgt = obj.labeler_.currTarget ;
        obj.tv_.updateTrackResI(xy, tfeo, iTgt) ;
      end
    end  % function

    function tfKPused = kpf(obj, ~, evt)
      % Handle key press.

      mdl = obj.model_ ;
      if ~obj.labeler_.isReady
        tfKPused = false ;
        return ;
      end

      key = evt.Key ;
      modifier = evt.Modifier ;
      tfCtrl = ismember('control', modifier) ;
      tfShft = any(strcmp('shift', modifier)) ;

      tfKPused = true ;
      lObj = obj.labeler_ ;
      lc = obj.labelerController_ ;
      if tfShft && strcmp(key, 'a')
        camroll(obj.hAx_(1), 2) ;
      elseif tfShft && strcmp(key, 'd')
        camroll(obj.hAx_(1), -2) ;
      elseif strcmp(key, 'w') && tfCtrl
        mdl.cbkNewTgt() ;
        obj.newPrimaryTarget() ;
      elseif strcmp(key, 'z') && tfCtrl
        mdl.undoLastLabel() ;
      elseif strcmp(key, 'o') && ~tfCtrl
        [tfSel, iSel] = mdl.anyPointSelected() ;
        if tfSel
          mdl.toggleEstOccPoint(iSel) ;
        end
        if mdl.state == LabelState.ACCEPTED
          mdl.storeLabels() ;
        end
      elseif any(strcmp(key, {'d' 'equal'})) && ~tfCtrl
        lc.frameUp(tfCtrl) ;
      elseif any(strcmp(key, {'a' 'hyphen'})) && ~tfCtrl
        lc.frameDown(tfCtrl) ;
      elseif ~tfCtrl && any(strcmp(key, {'leftarrow' 'rightarrow' 'uparrow' 'downarrow'}))
        [tfSel, iSel] = mdl.anyPointSelected() ;
        if tfSel
          xy = mdl.getLabelCoordsI(iSel) ;
          switch key
            case 'leftarrow'
              dxdy = -lc.videoCurrentRightVec() ;
            case 'rightarrow'
              dxdy = lc.videoCurrentRightVec() ;
            case 'uparrow'
              dxdy = lc.videoCurrentUpVec() ;
            case 'downarrow'
              dxdy = -lc.videoCurrentUpVec() ;
            otherwise
              error('LabelCoreSeqMAController:unknownKey', ...
                    'Unknown arrow key %s.', key) ;
          end
          if tfShft
            xyNew = xy + dxdy * 10 ;
          else
            xyNew = xy + dxdy ;
          end
          xyNew = lc.videoClipToVideo(xyNew) ;
          mdl.xy(iSel, :) = xyNew ;
          mdl.lastChangedIPt = iSel ;
          mdl.notify('updateLabelCoordsI') ;
        else
          tfKPused = false ;
        end
      elseif strcmp(key, 'backquote')
        iPt = mdl.kpfIPtFor1Key + 10 ;
        if iPt > mdl.nPts
          iPt = 1 ;
        end
        mdl.kpfIPtFor1Key = iPt ;
        obj.refreshTxLabelCoreAux() ;
      elseif any(strcmp(key, {'0' '1' '2' '3' '4' '5' '6' '7' '8' '9'}))
        if mdl.state ~= LabelState.LABEL
          iPt = str2double(key) ;
          if iPt == 0
            iPt = 10 ;
          end
          iPt = iPt + mdl.kpfIPtFor1Key - 1 ;
          if iPt > mdl.nPts
            return ;
          end
          mdl.toggleSelectPoint(iPt) ;
        end
      else
        tfKPused = false ;
      end
    end  % function

  end  % methods

  %% Two-click align (graphics)
  methods

    function tcInitGraphics(obj)
      % Create or reset the two-click graphical handles.
      if ~isempty(obj.tcHpts_)
        set(obj.tcHpts_, 'XData', nan, 'YData', nan) ;
      else
        obj.tcHpts_ = plot(obj.hAx_(1), nan, nan) ;
        set(obj.tcHpts_, obj.tcHptsPV_) ;
      end
    end  % function

    function hlpAxBDFTwoClick(obj, xy)
      % Handle two-click axis click: place marker, compute zoom/roll, delegate to model.

      mdl = obj.model_ ;
      h = obj.tcHpts_ ;
      switch mdl.tcipt_
        case 0
          set(h, 'XData', xy(1), 'YData', xy(2)) ;
          mdl.hlpAxBDFTwoClick(xy) ;
        case 1
          x0 = h.XData ;
          y0 = h.YData ;
          set(h, 'XData', [x0 xy(1)], 'YData', [y0 xy(2)]) ;
          mdl.hlpAxBDFTwoClick(xy) ;

          xc = (x0 + xy(1)) / 2 ;
          yc = (y0 + xy(2)) / 2 ;
          dx = x0 - xy(1) ;
          dy = y0 - xy(2) ;
          th = atan2(dy, dx) ;
          lc = obj.labelerController_ ;
          mdl.tc_prev_axis_ = lc.videoCurrentAxis() ;
          lc.videoCenterOnCurrTarget(xc, yc, th) ;
          rad = 2 * sqrt(dx.^2 + dy.^2) ;
          lc.videoZoom(rad) ;
          if ~mdl.tcShow_
            set(h, 'XData', nan, 'YData', nan) ;
          end
        otherwise
          error('LabelCoreSeqMAController:badTcipt', ...
                'Unexpected tcipt_ value %d.', mdl.tcipt_) ;
      end
    end  % function

    function onUpdateTwoClickState(obj) %#ok<MANU>
      % Respond to two-click state changes in the model.
      % Currently the graphics are handled inline by hlpAxBDFTwoClick.
    end  % function

  end  % methods

  %% MA buttons
  methods

    function addMAbuttons(obj)
      % Create New Target and Remove Target pushbuttons.

      mdl = obj.model_ ;
      btn = obj.pbRoiNew_ ;
      YOFF_NORMALIZED = .01 ;
      pos = btn.Position ;
      pos(2) = pos(2) + pos(4) + YOFF_NORMALIZED ;

      pb = uicontrol( ...
        'parent', obj.hFig_(1), ...
        'style', 'pushbutton', ...
        'units', btn.Units, ...
        'position', pos, ...
        'fontunits', btn.FontUnits, ...
        'fontsize', btn.FontSize, ...
        'fontweight', btn.FontWeight, ...
        'backgroundcolor', mdl.CLR_NEW_TGT, ...
        'string', 'New Target', ...
        'callback', @(s,e)obj.cbkNewTgt() ...
      ) ;
      obj.pbNewTgt_ = pb ;

      btn = obj.pbRoiEdit_ ;
      pos = btn.Position ;
      pos(2) = pos(2) + pos(4) + YOFF_NORMALIZED ;
      pb = uicontrol( ...
        'parent', obj.hFig_(1), ...
        'style', 'pushbutton', ...
        'units', btn.Units, ...
        'position', pos, ...
        'fontunits', btn.FontUnits, ...
        'fontsize', btn.FontSize, ...
        'fontweight', btn.FontWeight, ...
        'backgroundcolor', mdl.CLR_DEL_TGT, ...
        'string', 'Remove Target', ...
        'callback', @(s,e)obj.cbkDelTgt() ...
      ) ;
      obj.pbDelTgt_ = pb ;
    end  % function

    function cbkNewTgt(obj)
      % Callback for New Target button press.
      obj.model_.cbkNewTgt() ;
      obj.newPrimaryTarget() ;
    end  % function

    function cbkDelTgt(obj)
      % Callback for Delete Target button press.
      obj.model_.cbkDelTgt() ;
    end  % function

  end  % methods

  %% Controls and primary target
  methods

    function enableControls(obj)
      % Enable or disable buttons based on model state.

      mdl = obj.model_ ;
      if mdl.state == LabelState.LABEL
        set(obj.pbNewTgt_, 'Enable', 'on') ;
        set(obj.pbDelTgt_, 'Enable', 'off') ;
        set(obj.pbRoiNew_, 'Enable', 'off') ;
        set(obj.pbRoiEdit_, 'Enable', 'off') ;
        set(obj.pbNewTgt_, 'String', 'Cancel') ;
        obj.labelerController_.currImHud.hTxtTgt.BackgroundColor = mdl.CLR_NEW_TGT ;
      else
        set(obj.pbNewTgt_, 'Enable', 'on') ;
        set(obj.pbDelTgt_, 'Enable', 'on') ;
        set(obj.pbRoiNew_, 'Enable', 'on') ;
        set(obj.pbRoiEdit_, 'Enable', 'on') ;
        set(obj.pbNewTgt_, 'String', 'New Target') ;
        obj.labelerController_.currImHud.hTxtTgt.BackgroundColor = [0 0 0] ;
      end
    end  % function

    function newPrimaryTarget(obj)
      % Update which target is hidden in the multi-target visualizer.
      % The 'primary target' for LabelCoreSeqMA always matches lObj.currTarget.

      iTgt = obj.labeler_.currTarget ;
      if iTgt == 0
        iTgt = [] ; % ie dont hide any targets
      end
      obj.tv_.updateHideTarget(iTgt) ;
    end  % function

  end  % methods

  %% ROI methods
  methods

    function roiInit(obj)
      % Initialize the ROI rectangle drawer.
      obj.roiRectDrawer_ = RectDrawer(obj.hAx_(1)) ;
      obj.roiSetShow(false) ;
    end  % function

    function roiAddButtons(obj)
      % Create ROI-related pushbuttons (New Label Box, Edit Label Boxes).

      btn = obj.tbAccept_ ;
      pb = uicontrol( ...
        'parent', obj.hFig_(1), ...
        'style', 'pushbutton', ...
        'units', btn.Units, ...
        'position', btn.Position, ...
        'fontunits', btn.FontUnits, ...
        'fontsize', btn.FontSize, ...
        'fontweight', btn.FontWeight, ...
        'backgroundcolor', obj.CLR_PBROINEW_, ...
        'string', 'New Label Box', ...
        'units', btn.Units, ...
        'callback', @(s,e)obj.cbkRoiNew() ...
      ) ;
      obj.pbRoiNew_ = pb ;

      btn = obj.pbClear_ ;
      pb = uicontrol( ...
        'parent', obj.hFig_(1), ...
        'style', 'togglebutton', ...
        'units', btn.Units, ...
        'position', btn.Position, ...
        'fontunits', btn.FontUnits, ...
        'fontsize', btn.FontSize, ...
        'fontweight', btn.FontWeight, ...
        'backgroundcolor', obj.CLR_PBROIEDIT_, ...
        'string', 'Edit Label Boxes', ...
        'units', btn.Units, ...
        'callback', @(s,e)obj.cbkRoiEdit() ...
      ) ;
      obj.pbRoiEdit_ = pb ;
    end  % function

    function roiSetShow(obj, tf)
      % Show or hide ROI buttons and drawn rectangles.

      onoff = onIff(tf) ;
      obj.pbRoiEdit_.Visible = onoff ;
      obj.pbRoiNew_.Visible = onoff ;
      obj.roiRectDrawer_.setShowRois(tf) ;
      if tf
        lObj = obj.labeler_ ;
        if ~lObj.isinit && lObj.hasMovie
          frm = lObj.currFrame ;
          vroi = lObj.labelroiGet(frm) ;
          obj.roiRectDrawer_.setRois(vroi) ;
          obj.roiUpdatePBEdit(false) ;
        end
      end
    end  % function

    function cbkRoiNew(obj)
      % Callback for New Label Box button.

      assert(obj.labeler_.showMaRoiAux) ;
      set(obj.pbNewTgt_, 'Enable', 'off') ;
      set(obj.pbDelTgt_, 'Enable', 'off') ;
      set(obj.pbRoiNew_, 'Enable', 'off') ;
      set(obj.pbRoiEdit_, 'Enable', 'off') ;
      obj.roiRectDrawer_.newRoiDraw() ;
      v = obj.roiRectDrawer_.getRoisVerts() ;
      obj.labeler_.labelroiSet(v) ;
      set(obj.pbNewTgt_, 'Enable', 'on') ;
      set(obj.pbDelTgt_, 'Enable', 'on') ;
      set(obj.pbRoiNew_, 'Enable', 'on') ;
      set(obj.pbRoiEdit_, 'Enable', 'on') ;
    end  % function

    function cbkRoiEdit(obj)
      % Callback for Edit Label Boxes togglebutton.

      tfEditingNew = obj.pbRoiEdit_.Value ;
      rrd = obj.roiRectDrawer_ ;
      rrd.setEdit(tfEditingNew) ;
      if ~tfEditingNew
        v = rrd.getRoisVerts() ;
        obj.labeler_.labelroiSet(v) ;
      end
      obj.roiUpdatePBEdit(tfEditingNew) ;
    end  % function

    function roiUpdatePBEdit(obj, tf)
      % Update the Edit Label Boxes button text and value.
      if tf
        str = 'Done Editing' ;
        val = 1 ;
      else
        str = 'Edit Label Boxes' ;
        val = 0 ;
      end
      set(obj.pbRoiEdit_, 'Value', val, 'String', str) ;
    end  % function

  end  % methods

  %% Show/hide viz overrides
  methods

    function updateLabelVisibility(obj)
      % Hide/show labels state changed.
      doShowLabels = obj.labeler_.doShowLabels ;
      obj.tv_.setHideViz(~doShowLabels) ;
      updateLabelVisibility@LabelCoreController(obj) ;
    end
  end  % methods

  %% Cosmetics overrides
  methods

    function updateSkeletonEdges(obj, varargin)
      % Rebuild skeleton edge handles and update TV.
      updateSkeletonEdges@LabelCoreController(obj, varargin{:}) ;
      se = obj.model_.skeletonEdges() ;
      if ~isempty(obj.tv_)
        obj.tv_.initAndUpdateSkeletonEdges(se) ;
      end
    end  % function

    function updateShowSkeleton(obj)
      % Show or hide skeleton edges and update TV.
      updateShowSkeleton@LabelCoreController(obj) ;
      if ~isempty(obj.tv_)      
        tf = obj.labeler_.showSkeleton ;
        obj.tv_.setShowSkeleton(tf) ;
      end
    end  % function

    function updateColors(obj, colors)
      % Update colors for point markers, text labels, and TV.
      updateColors@LabelCoreController(obj, colors) ;
      obj.tv_.updateLandmarkColors(colors) ;
    end  % function

    function updateMarkerCosmetics(obj, pvMarker)
      % Update marker cosmetics for all point handles and TV.
      updateMarkerCosmetics@LabelCoreController(obj, pvMarker) ;
      obj.tv_.setMarkerCosmetics(pvMarker) ;
    end  % function

    function updateTextLabelCosmetics(obj, pvText, txtoffset)
      % Update text label cosmetics and TV.
      updateTextLabelCosmetics@LabelCoreController(obj, pvText, txtoffset) ;
      obj.tv_.setTextCosmetics(pvText) ;
      obj.tv_.setTextOffset(txtoffset) ;
    end  % function

    function skeletonCosmeticsUpdated(obj)
      % Refresh skeleton edge cosmetics from labeler and TV.
      skeletonCosmeticsUpdated@LabelCoreController(obj) ;
      obj.tv_.skeletonCosmeticsUpdated() ;
    end  % function

    function preProcParamsChanged(obj)
      % React to preproc param mutation on labeler.
      obj.tv_.updatePches() ;
    end  % function

  end  % methods

  %% Presentation
  methods

    function shortcuts = LabelShortcuts(obj)
      % Return shortcut descriptions for SeqMA mode.

      lc = obj.labelerController_ ;
      shortcuts = cell(0, 3) ;

      shortcuts{end+1, 1} = 'New target' ;
      shortcuts{end, 2} = 'w' ;
      shortcuts{end, 3} = {'Ctrl'} ;

      shortcuts{end+1, 1} = 'Undo last label click' ;
      shortcuts{end, 2} = 'z' ;
      shortcuts{end, 3} = {'Ctrl'} ;

      shortcuts{end+1, 1} = 'Toggle whether selected kpt is occluded' ;
      shortcuts{end, 2} = 'o' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Rotate axes CCW by 2 degrees' ;
      shortcuts{end, 2} = 'A' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Rotate axes CW by 2 degrees' ;
      shortcuts{end, 2} = 'D' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Toggle whether panning tool is on' ;
      shortcuts{end, 2} = 'a' ;
      shortcuts{end, 3} = {'Ctrl'} ;

      shortcuts{end+1, 1} = 'Toggle whether panning tool is on' ;
      shortcuts{end, 2} = 'd' ;
      shortcuts{end, 3} = {'Ctrl'} ;

      shortcuts{end+1, 1} = 'Forward one frame' ;
      shortcuts{end, 2} = '= or d' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Backward one frame' ;
      shortcuts{end, 2} = '- or a' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Un/Select kpt of current target' ;
      shortcuts{end, 2} = '0-9' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Toggle which kpts 0-9 correspond to' ;
      shortcuts{end, 2} = '`' ;
      shortcuts{end, 3} = {} ;

      rightpx = lc.videoCurrentRightVec() ;
      rightpx = rightpx(1) ;
      uppx = lc.videoCurrentUpVec() ;
      uppx = abs(uppx(2)) ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move right by %.1f px, ow forward one frame', rightpx) ;
      shortcuts{end, 2} = 'Right arrow' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move left by %.1f px, ow back one frame', rightpx) ;
      shortcuts{end, 2} = 'Left arrow' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move up by %.1f px', uppx) ;
      shortcuts{end, 2} = 'Up arrow' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move down by %.1f px', uppx) ;
      shortcuts{end, 2} = 'Down arrow' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move right by %.1f px', 10*rightpx) ;
      shortcuts{end, 2} = '+' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move left by %.1f px', 10*rightpx) ;
      shortcuts{end, 2} = '-' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = ...
        sprintf('If kpt selected, move left by %.1f px, ow go to next %s', ...
                10*rightpx, obj.labeler_.movieShiftArrowNavMode.prettyStr) ;
      shortcuts{end, 2} = 'Left arrow' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = ...
        sprintf('If kpt selected, move right by %.1f px, ow go to previous %s', ...
                10*rightpx, obj.labeler_.movieShiftArrowNavMode.prettyStr) ;
      shortcuts{end, 2} = 'Right arrow' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move up by %.1f px', 10*uppx) ;
      shortcuts{end, 2} = 'Up arrow' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move down by %.1f px', 10*uppx) ;
      shortcuts{end, 2} = 'Down arrow' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = 'Zoom in/out' ;
      shortcuts{end, 2} = 'Mouse scroll' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Pan view' ;
      shortcuts{end, 2} = 'Mouse right-click-drag' ;
      shortcuts{end, 3} = {} ;
    end  % function

    function h = getLabelingHelp(obj)
      % Return labeling help text for SeqMA mode.

      mdl = obj.model_ ;
      h = { ...
        'To{\bf add a target}: '
        ' - Push the New Target button.'
        } ;
      if mdl.tcOn_
        h{end+1} = ' - Click two points on the new target to zoom in on it.' ;
        h{end+1} = '   Often, these points correspond to the animal''s head and tail.' ;
      end
      h{end+1} = ' - Click the locations of your keypoints in order.' ;
      h{end+1} = ' - Hold shift while clicking to annotate that a keypoint is occluded.' ;
      h{end+1} = ' - You do not need to label all animals in each frame. ' ;
      h{end+1} = '   the black boxes show regions of the image around your labeled' ;
      h{end+1} = '   animals. APT only uses these boxes for training. If another' ;
      h{end+1} = '   animal is inside one of your label boxes, you should label it.' ;
      h{end+1} = '' ;
      h{end+1} = 'Use{\bf Label Boxes} to specify image regions that are completely labeled. ' ;
      h{end+1} = '  This is important for teaching the classifier what a negative label is. ' ;
      h{end+1} = '  An image region is completely labeled if no keypoints in that region' ;
      h{end+1} = '  are unlabeled. You e.g. can draw a label box around parts of the image' ;
      h{end+1} = '  that do not contain animals to add negative training examples.' ;
      h{end+1} = ' - Click New Label Box to add a new label box.' ;
      h{end+1} = '' ;
      h{end+1} = 'To{\bf set zoom}, at any time, mouse-scroll to zoom and' ;
      h{end+1} = '  right-click-drag to pan.' ;
      h{end+1} = '  Type Ctrl + f to zoom out and show the full frame.' ;
      h{end+1} = '' ;
      h{end+1} = 'To{\bf adjust labeled keypoints}:' ;
      h{end+1} = ' - Select the corresponding target number from the "Targets" box. ' ;
      h{end+1} = ' - Click the point or type its number to select a point. ' ;
      h{end+1} = '   Once selected, click the new location or use the arrow keys' ;
      h{end+1} = '   to move it. ' ;
      h{end+1} = ' - Alternatively, you can click and drag the keypoint.' ;
      h{end+1} = '' ;
      h{end+1} = 'To{\bf edit Label Boxes}:' ;
      h{end+1} = ' - Click Edit Label Boxes to enable editing. ' ;
      h{end+1} = ' - Drag the corners of a box to move or resize it.' ;
      h{end+1} = ' - Right-click the box and select Remove Rectangle to delete it.' ;
      h{end+1} = ' - Re-click Edit Label Boxes to register your changes.' ;
      h{end+1} = '' ;

      h1 = getLabelingHelp@LabelCoreController(obj) ;
      h = [h(:) ; h1(:)] ;
    end  % function

  end  % methods

end  % classdef
