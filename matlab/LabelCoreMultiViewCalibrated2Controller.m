classdef LabelCoreMultiViewCalibrated2Controller < LabelCoreController
% Multiview calibrated labeling controller
%
% Owns all graphics for multi-view calibrated labeling: point handles in
% multiple views, epipolar line handles, reconstructed point handles, 3D
% display handles, axis xlabels. Receives GUI callbacks, extracts GUI
% state (mouse position, axis, modifiers), delegates data logic to
% LabelCoreMultiViewCalibrated2Model, and syncs graphics in response to
% model events.
%
% This is the controller half of the LabelCoreMultiViewCalibrated2 MVC
% split. The model half is LabelCoreMultiViewCalibrated2Model.

  properties
    supportsSingleView = false ;
    supportsMultiView = true ;
    supportsCalibration = true ;
    supportsMultiAnimal = false ;
  end

  properties (Transient)
    hPtsTxtStrs_              % [npts] cellstr, text labels for each pt
    pjtHLinesEpi_             % [nview x nview] line handles for epipolar lines
    pjtHLinesRecon_           % [nview] line handles for reconstructed pts
    pjtShow3D_                % [nShow3D] aux handles for showing 3D info
    hAxXLabels_               % [nview] xlabel handles
    hAxXLabelsFontSize_ = 11 ;
  end

  methods

    function obj = LabelCoreMultiViewCalibrated2Controller(labelerController, labeler, model)
      % Construct a LabelCoreMultiViewCalibrated2Controller.
      obj = obj@LabelCoreController(labelerController, labeler, model) ;
    end  % function

    function delete(obj)
      % Clean up projection graphics handles.
      deleteValidGraphicsHandles(obj.pjtHLinesEpi_) ;
      obj.pjtHLinesEpi_ = [] ;
      deleteValidGraphicsHandles(obj.pjtHLinesRecon_) ;
      obj.pjtHLinesRecon_ = [] ;
      deleteValidGraphicsHandles(obj.hAxXLabels_) ;
      obj.hAxXLabels_ = [] ;
      deleteValidGraphicsHandles(obj.pjtShow3D_) ;
      obj.pjtShow3D_ = [] ;
    end  % function

    function initHook(obj)
      % Initialize multi-view specific graphics: point handles per-view,
      % epipolar/recon line handles, xlabels, and model event listeners.

      mdl = obj.model_ ;

      % Redefine .hPts_, .hPtsTxt_ for multi-view (originally initted
      % in LabelCoreController.init() for single-axis only)
      deleteValidGraphicsHandles(obj.hPts_) ;
      deleteValidGraphicsHandles(obj.hPtsTxt_) ;
      deleteValidGraphicsHandles(obj.hSkel_) ;
      obj.hPts_ = gobjects(mdl.nPts, 1) ;
      obj.hPtsTxt_ = gobjects(mdl.nPts, 1) ;
      obj.hSkel_ = gobjects(size(mdl.skeletonEdges(), 1), 1) ;

      ppi = mdl.ptsPlotInfo ;
      obj.hPtsTxtStrs_ = cell(mdl.nPts, 1) ;

      obj.updateSkeletonEdges() ;
      obj.updateShowSkeleton() ;

      pvMarker = struct2paramscell(ppi.MarkerProps) ;
      pvText = struct2paramscell(ppi.TextProps) ;

      for iPt = 1:mdl.nPts
        iSet = mdl.iPt2iSet_(iPt) ;
        setClr = ppi.Colors(iSet, :) ;
        ptsArgs = {nan, nan, pvMarker{:}, ...
          'ZData', 1, ...
          'Color', setClr, ...
          'UserData', iPt, ...
          'HitTest', 'on', ...
          'ButtonDownFcn', @(s,e)obj.ptBDF(s, e)} ; %#ok<CCAT>
        ax = obj.hAx_(mdl.iPt2iAx_(iPt)) ;
        obj.hPts_(iPt) = plot(ax, ptsArgs{:}, ...
          'Tag', sprintf('LabelCoreMV_Pt%d', iPt)) ;
        txtStr = num2str(iSet) ;
        txtargs = {'Color', setClr, pvText{:}, 'PickableParts', 'none'} ; %#ok<CCAT>
        obj.hPtsTxt_(iPt) = text(nan, nan, txtStr, ...
          'Parent', ax, txtargs{:}, ...
          'Tag', sprintf('LabelCoreMV_PtTxt%d', iPt)) ;
        obj.hPtsTxtStrs_{iPt} = txtStr ;
      end

      % Sync initial coords from model
      obj.updateLabelCoords() ;

      % Axis xlabels
      obj.hAxXLabels_ = gobjects(mdl.nView, 1) ;
      for iView = 1:mdl.nView
        ax = obj.hAx_(iView) ;
        obj.hAxXLabels_(iView) = xlabel(ax, '', 'fontsize', obj.hAxXLabelsFontSize_) ;
      end
      obj.txLblCoreAux_.Visible = 'on' ;
      obj.refreshHotkeyDesc() ;

      obj.labeler_.currImHudModel.hasLblPt = true ;
      obj.labeler_.notify('updateHudReadoutFields') ;

      % Set up axis BDFs for multi-view
      for iView = 1:mdl.nView
        set(obj.hAx_(iView), 'ButtonDownFcn', @(s,e)obj.axBDF(s, e)) ;
      end

      % Initialize projection graphics
      obj.projectionWorkingSetClear_() ;
      obj.projectionInit_() ;

      % Register model listeners for multi-view specific events
      obj.listeners_ = [ ...
        obj.listeners_ ; ...
        addlistener(mdl, 'updateAdjusted',    @(s,e)obj.onUpdateAdjusted()) ; ...
        addlistener(mdl, 'updateProjection',  @(s,e)obj.onUpdateProjection()) ; ...
        addlistener(mdl, 'updateWorkingSet',  @(s,e)obj.onUpdateWorkingSet()) ; ...
      ] ;
    end  % function

  end  % methods

  %% Occluded box override (multi-view)
  methods

    function showOccHook(obj)
      % Create occluded-box point handles per-view for multi-view.
      mdl = obj.model_ ;
      ppi = mdl.ptsPlotInfo ;

      deleteValidGraphicsHandles(obj.hPtsOcc_) ;
      deleteValidGraphicsHandles(obj.hPtsTxtOcc_) ;
      obj.hPtsOcc_ = gobjects(mdl.nPts, 1) ;
      obj.hPtsTxtOcc_ = gobjects(mdl.nPts, 1) ;

      pvMarker = struct2paramscell(ppi.MarkerProps) ;
      pvText = struct2paramscell(ppi.TextProps) ;

      for iPt = 1:mdl.nPts
        iSet = mdl.iPt2iSet_(iPt) ;
        setClr = ppi.Colors(iSet, :) ;
        ptsArgsOcc = {nan, nan, pvMarker{:}, ...
          'Color', setClr, ...
          'UserData', iPt, ...
          'HitTest', 'off'} ; %#ok<CCAT>
        axocc = obj.hAxOcc_(mdl.iPt2iAx_(iPt)) ;
        obj.hPtsOcc_(iPt) = plot(axocc, ptsArgsOcc{:}, ...
          'Tag', sprintf('LabelCoreMV_PtOcc%d', iPt)) ;
        txtStr = num2str(iSet) ;
        txtargs = {'Color', setClr, ...
          pvText{:}, ...
          'PickableParts', 'none'} ; %#ok<CCAT>
        obj.hPtsTxtOcc_(iPt) = text(nan, nan, txtStr, ...
          'Parent', axocc, txtargs{:}, ...
          'Tag', sprintf('LabelCoreMV_PtTxtOcc%d', iPt)) ;
      end

      for iVw = 1:mdl.nView
        axis(obj.hAxOcc_(iVw), [0 mdl.nPointSet+1 0 2]) ;
      end
    end  % function

  end  % methods

  %% Skeleton overrides (multi-view)
  methods

    function updateSkeletonEdges(obj)
      % Rebuild skeleton edge handles for multi-view.
      mdl = obj.model_ ;

      if isempty(mdl.iSet2iPt_) || isempty(obj.labeler_.skeletonEdges)
        return ;
      end

      ax = obj.hAx_ ;
      ptsPlotInfo = mdl.ptsPlotInfo ;
      edges = mdl.skeletonEdges() ;

      deleteValidGraphicsHandles(obj.hSkel_) ;
      nEdgesPerView = size(edges, 1) / mdl.nView ;
      obj.hSkel_ = gobjects(size(edges, 1), 1) ;
      for ivw = 1:mdl.nView
        for i = 1:size(obj.labeler_.skeletonEdges, 1)
          iEdge = (ivw - 1)*nEdgesPerView + i ;
          obj.hSkel_(iEdge) = LabelCoreController.initSkeletonEdge(ax(ivw), iEdge, ptsPlotInfo) ;
        end
      end
      xy = mdl.getLabelCoords() ;
      tfOccld = any(isinf(xy), 2) ;
      LabelCoreController.setSkelCoords(xy, tfOccld, obj.hSkel_, edges) ;
    end  % function

  end  % methods

  %% Occluded pts override (multi-view, set-based layout)
  methods

    function refreshOccludedPts(obj)
      % Based on model.tfOcc_: hide occluded points in main image; arrange
      % occluded points in occluded box, positioned by set index.
      mdl = obj.model_ ;

      if isempty(obj.hPtsOcc_)
        return ;
      end

      tf = mdl.tfOcc ;
      assert(isvector(tf) && numel(tf) == mdl.nPts) ;
      nOcc = nnz(tf) ;
      iOcc = find(tf) ;
      obj.setPtsCoords(nan(nOcc, 2), obj.hPts_(tf), obj.hPtsTxt_(tf)) ;
      for iPt = iOcc(:)'
        iSet = mdl.iPt2iSet_(iPt) ;
        setPositionsOfLabelLinesAndTextsBangBang( ...
          obj.hPtsOcc_(iPt), obj.hPtsTxtOcc_(iPt), [iSet 1], 0.25) ;
      end
      setPositionsOfLabelLinesAndTextsBangBang( ...
        obj.hPtsOcc_(~tf), obj.hPtsTxtOcc_(~tf), nan(mdl.nPts - nOcc, 2), 0.25) ;
    end  % function

  end  % methods

  %% Model event handlers
  methods

    function onUpdateState(obj)
      % Sync tbAccept appearance to model state.
      mdl = obj.model_ ;
      switch mdl.state
        case LabelState.ADJUST
          set(obj.tbAccept_, 'BackgroundColor', [0.6, 0, 0], 'String', 'Accept', ...
            'Value', 0, 'Enable', 'on') ;
        case LabelState.ACCEPTED
          set(obj.tbAccept_, 'BackgroundColor', [0, 0.4, 0], 'String', 'Accepted', ...
            'Value', 1, 'Enable', 'on') ;
        otherwise
          error('LabelCoreMultiViewCalibrated2Controller:unknownState', ...
            'Unknown state %s.', char(mdl.state)) ;
      end
    end  % function

    function onUpdateAdjusted(obj)
      % Sync point colors for adjusted/unadjusted points.
      mdl = obj.model_ ;
      ppi = mdl.ptsPlotInfo ;
      iPt = mdl.lastChangedIPt ;
      if iPt == 0
        % All points changed
        if all(mdl.tfAdjusted_)
          % All adjusted: show in color
          clrs = ppi.Colors ;
          for i = 1:mdl.nPts
            set(obj.hPts_(i), 'Color', clrs(i, :)) ;
            if ~isempty(obj.hPtsOcc_)
              set(obj.hPtsOcc_(i), 'Color', clrs(i, :)) ;
            end
          end
        else
          % All unadjusted: show in template color or hide
          if mdl.streamlined_
            [obj.hPts_.XData] = deal(nan) ;
            [obj.hPts_.YData] = deal(nan) ;
            [obj.hPtsTxt_.Position] = deal([nan nan 1]) ;
          else
            tpClr = ppi.TemplateMode.TemplatePointColor ;
            obj.refreshPtMarkers() ;
            arrayfun(@(x)set(x, 'Color', tpClr), obj.hPts_) ;
          end
        end
      else
        % Single point changed
        if mdl.tfAdjusted_(iPt)
          clr = ppi.Colors(iPt, :) ;
          set(obj.hPts_(iPt), 'Color', clr) ;
          if ~isempty(obj.hPtsOcc_)
            set(obj.hPtsOcc_(iPt), 'Color', clr) ;
          end
        end
      end
    end  % function

    function onUpdateProjection(obj)
      % Projection state changed. Clear and refresh epipolar/recon lines.
      obj.projectionClear_() ;
      obj.projectionRefresh_() ;
    end  % function

    function onUpdateWorkingSet(obj)
      % Working set changed. Update point boldness/dimness and xlabels.
      mdl = obj.model_ ;
      iWS = mdl.iSetWorking_ ;

      if isnan(iWS)
        % Working set cleared
        obj.projectionWorkingSetClear_() ;
      else
        % Working set selected
        obj.projectionWorkingSetSet_(iWS) ;
      end
    end  % function

  end  % methods

  %% GUI callback handlers
  methods

    function axBDF(obj, src, evt)
      % Handle axis button-down: identify which axis, jump working set
      % point to click location.
      mdl = obj.model_ ;
      if ~obj.labeler_.isReady
        return ;
      end

      if evt.Button ~= 1
        return ;
      end

      iAx = find(src == obj.hAx_) ;

      if obj.isPanZoom(iAx)
        return ;
      end

      iWS = mdl.iSetWorking_ ;
      if ~isnan(iWS)
        iPt = mdl.iSet2iPt_(iWS, iAx) ;
        ax = obj.hAx_(iAx) ;
        pos = get(ax, 'CurrentPoint') ;
        pos = pos(1, 1:2) ;
        mdl.xy(iPt, :) = pos ;
        mdl.lastChangedIPt = iPt ;
        mdl.notify('updateLabelCoordsI') ;
        mdl.setPointAdjusted(iPt) ;

        if mdl.tfOcc(iPt)
          mdl.tfOcc(iPt) = false ;
          obj.refreshOccludedPts() ;
        end

        if mdl.streamlined_ && all(mdl.tfAdjusted_)
          mdl.enterAccepted(true) ;
        else
          switch mdl.state
            case LabelState.ADJUST
              % none
            case LabelState.ACCEPTED
              mdl.enterAdjust(false, false) ;
            otherwise
              error('LabelCoreMultiViewCalibrated2Controller:unknownState', ...
                'Unknown state %s.', char(mdl.state)) ;
          end
        end
        obj.projectionRefresh_() ;
      end
    end  % function

    function axOccBDF(obj, src, evt) %#ok<INUSD>
      % Handle occluded-axis button-down.
      mdl = obj.model_ ;
      if ~obj.labeler_.isReady
        return ;
      end

      iAx = find(src == obj.hAxOcc_) ;
      if obj.isPanZoom(iAx)
        return ;
      end

      assert(isscalar(iAx)) ;
      iWS = mdl.iSetWorking_ ;
      if ~isnan(iWS)
        iPt = mdl.iSet2iPt_(iWS, iAx) ;
        obj.setPtFullOcc_(iPt) ;
      end
    end  % function

    function ptBDF(obj, src, evt)
      % Handle point button-down: initiate drag or toggle est-occ.
      mdl = obj.model_ ;
      if ~obj.labeler_.isReady
        return ;
      end
      ax = get(src, 'Parent') ;
      iAx = find(ax == obj.hAx_) ;
      if obj.isPanZoom(iAx)
        return ;
      end

      iPt = src.UserData ;
      switch evt.Button
        case 1
          mdl.iPtMove_ = iPt ;
          mdl.tfMoved_ = false ;
        case 3
          mdl.toggleEstOccPoint(iPt) ;
        otherwise
          % ignore other buttons
      end
    end  % function

    function wbmf(obj, src, evt) %#ok<INUSD>
      % Handle window button motion: drag point.
      mdl = obj.model_ ;
      if ~obj.labeler_.isReady
        return ;
      end

      iPt = mdl.iPtMove_ ;
      if ~isnan(iPt)
        if mdl.state == LabelState.ACCEPTED
          mdl.enterAdjust(false, false) ;
        end

        iAx = mdl.iPt2iAx_(iPt) ;
        ax = obj.hAx_(iAx) ;
        tmp = get(ax, 'CurrentPoint') ;
        pos = tmp(1, 1:2) ;
        mdl.tfMoved_ = true ;
        mdl.xy(iPt, :) = pos ;
        mdl.lastChangedIPt = iPt ;
        mdl.notify('updateLabelCoordsI') ;
        mdl.setPointAdjusted(iPt) ;

        obj.projectionRefresh_() ;
      end
    end  % function

    function wbuf(obj, src, evt) %#ok<INUSD>
      % Handle window button up: end drag.
      mdl = obj.model_ ;
      if ~obj.labeler_.isReady
        return ;
      end

      iPt = mdl.iPtMove_ ;
      if ~isnan(iPt)
        obj.projectionRefresh_() ;
      end
      mdl.iPtMove_ = nan ;
      mdl.tfMoved_ = false ;
    end  % function

    function tfKPused = kpf(obj, src, evt)
      % Handle key press: accept, frame nav, arrow-adjust, working set select.
      mdl = obj.model_ ;
      if ~obj.labeler_.isReady
        tfKPused = false ;
        return ;
      end

      key = evt.Key ;
      modifier = evt.Modifier ;
      tfCtrl = any(strcmp('control', modifier)) ;

      tfKPused = true ;

      % shortcuts in main figure will be handled by menu items themselves
      if src == obj.labelerController_.mainFigure_
        match = [] ;
      else
        match = obj.labelerController_.matchShortcut(evt) ;
      end

      if strcmp(key, 'space')
        obj.toggleEpipolarState_() ;
      elseif strcmp(key, 's') && ~tfCtrl
        if mdl.state == LabelState.ADJUST
          mdl.acceptLabels() ;
        end
      elseif any(strcmp(key, {'d' 'equal'}))
        obj.labelerController_.frameUp(tfCtrl) ;
      elseif any(strcmp(key, {'a' 'hyphen'}))
        obj.labelerController_.frameDown(tfCtrl) ;
      elseif strcmp(key, 'o') && ~tfCtrl
        [tfSel, iSel] = mdl.projectionPointSelected() ;
        if tfSel
          mdl.toggleEstOccPoint(iSel) ;
        end
      elseif strcmp(key, 'u') && ~tfCtrl
        iAx = find(gcf == obj.hFig_) ;
        iWS = mdl.iSetWorking_ ;
        if isscalar(iAx) && ~isnan(iWS) && obj.labeler_.showOccludedBox
          iPt = mdl.iSet2iPt_(iWS, iAx) ;
          obj.setPtFullOcc_(iPt) ;
        end
      elseif any(strcmp(key, {'leftarrow' 'rightarrow' 'uparrow' 'downarrow'}))
        [tfSel, iSel, iAx] = mdl.projectionPointSelected() ;
        if tfSel && ~mdl.tfOcc(iSel)
          tfShift = any(strcmp('shift', modifier)) ;
          xy = mdl.getLabelCoordsI(iSel) ;
          ax = obj.hAx_(iAx) ;
          switch key
            case 'leftarrow'
              xl = xlim(ax) ;
              dx = diff(xl) ;
              if tfShift
                xy(1) = xy(1) - dx/mdl.DXFACBIG ;
              else
                xy(1) = xy(1) - dx/mdl.DXFAC ;
              end
            case 'rightarrow'
              xl = xlim(ax) ;
              dx = diff(xl) ;
              if tfShift
                xy(1) = xy(1) + dx/mdl.DXFACBIG ;
              else
                xy(1) = xy(1) + dx/mdl.DXFAC ;
              end
            case 'uparrow'
              yl = ylim(ax) ;
              dy = diff(yl) ;
              if tfShift
                xy(2) = xy(2) - dy/mdl.DXFACBIG ;
              else
                xy(2) = xy(2) - dy/mdl.DXFAC ;
              end
            case 'downarrow'
              yl = ylim(ax) ;
              dy = diff(yl) ;
              if tfShift
                xy(2) = xy(2) + dy/mdl.DXFACBIG ;
              else
                xy(2) = xy(2) + dy/mdl.DXFAC ;
              end
            otherwise
              error('LabelCoreMultiViewCalibrated2Controller:unknownKey', ...
                'Unknown arrow key %s.', key) ;
          end
          mdl.setLabelCoordsI(xy, iSel) ;
          switch mdl.state
            case LabelState.ADJUST
              mdl.setPointAdjusted(iSel) ;
            case LabelState.ACCEPTED
              mdl.enterAdjust(false, false) ;
            otherwise
              error('LabelCoreMultiViewCalibrated2Controller:unknownState', ...
                'Unknown state %s.', char(mdl.state)) ;
          end
          obj.projectionRefresh_() ;
        else
          tfKPused = false ;
        end
      elseif strcmp(key, 'backquote')
        iSet = mdl.numHotKeyPtSet_ + 10 ;
        if iSet > mdl.nPointSet
          iSet = 1 ;
        end
        mdl.numHotKeyPtSet_ = iSet ;
        obj.refreshHotkeyDesc() ;
      elseif any(strcmp(key, {'0' '1' '2' '3' '4' '5' '6' '7' '8' '9'}))
        iSet = str2double(key) ;
        if iSet == 0
          iSet = 10 ;
        end
        iSet = iSet + mdl.numHotKeyPtSet_ - 1 ;
        if iSet > mdl.nPointSet
          % none
        else
          tfIsClearOnly = iSet == mdl.iSetWorking_ ;
          mdl.projectionWorkingSetClear_() ;
          obj.projectionClear_() ;
          if ~tfIsClearOnly
            mdl.projectionWorkingSetSet_(iSet) ;
          end
        end
      elseif ~isempty(match)
        for i = 1:size(match, 1)
          tag = match{i, 1} ;
          cb = obj.labelerController_.(tag).Callback ;
          if ischar(cb)
            eval(cb) ;
          else
            cb(obj.labelerController_.(tag), evt) ;
          end
        end
      else
        tfKPused = false ;
      end
    end  % function

  end  % methods

  %% Internal helpers
  methods

    function setPtFullOcc_(obj, iPt)
      % Mark a point as fully occluded and update graphics.
      mdl = obj.model_ ;
      mdl.setPointAdjusted(iPt) ;

      mdl.tfOcc(iPt) = true ;
      mdl.tfEstOcc(iPt) = false ;
      obj.refreshOccludedPts() ;
      obj.refreshPtMarkers('iPts', iPt) ;

      if mdl.streamlined_ && all(mdl.tfAdjusted_)
        mdl.enterAccepted(true) ;
      else
        switch mdl.state
          case LabelState.ADJUST
            % none
          case LabelState.ACCEPTED
            mdl.enterAdjust(false, false) ;
          otherwise
            error('LabelCoreMultiViewCalibrated2Controller:unknownState', ...
              'Unknown state %s.', char(mdl.state)) ;
        end
      end
      obj.projectionRefresh_() ;
    end  % function

  end  % methods

  %% Projection graphics
  methods

    function projectionInit_(obj)
      % Create epipolar and reconstructed point line handles.
      mdl = obj.model_ ;
      ppimvcm = mdl.ptsPlotInfo.MultiViewCalibratedMode ;
      gdata = obj.labelerController_ ;
      nView = mdl.nView ;

      hLEpi = gobjects(nView, nView) ;
      hLRcn = gobjects(1, nView) ;
      for iV = 1:nView
        ax = gdata.axes_all(iV) ;
        for j = 1:nView
          hLEpi(iV, j) = plot(ax, nan, nan, '-', ...
            'LineWidth', ppimvcm.EpipolarLineWidth, ...
            'Marker', '.', ...
            'MarkerSize', 1, ...
            'PickableParts', 'none', ...
            'Tag', sprintf('LabelCoreMV_Epi%d', iV)) ;
        end
        hLRcn(iV) = plot(ax, nan, nan, ppimvcm.ReconstructedMarker, ...
          'MarkerSize', ppimvcm.ReconstructedMarkerSize, ...
          'LineWidth', ppimvcm.ReconstructedLineWidth, ...
          'PickableParts', 'none', ...
          'Tag', sprintf('LabelCoreMV_Rcn%d', iV)) ;
      end
      obj.pjtHLinesEpi_ = hLEpi ;
      obj.pjtHLinesRecon_ = hLRcn ;
    end  % function

    function projectionClear_(obj)
      % Clear projection state and line graphics.
      mdl = obj.model_ ;

      for iPt = 1:mdl.nPts
        set(obj.hPtsTxt_(iPt), 'String', obj.hPtsTxtStrs_{iPt}) ;
      end

      mdl.showEpiLines_ = true ;

      for i = 1:mdl.nView
        for j = 1:mdl.nView
          set(obj.pjtHLinesEpi_(i, j), 'XData', nan, 'YData', nan, 'Visible', 'off') ;
        end
      end

      set(obj.pjtHLinesRecon_, 'Visible', 'off') ;
    end  % function

    function toggleEpipolarState_(obj)
      % Toggle epipolar line visibility.
      mdl = obj.model_ ;
      if mdl.showEpiLines_
        mdl.showEpiLines_ = false ;
        set(obj.pjtHLinesEpi_, 'Visible', 'off') ;
      else
        mdl.showEpiLines_ = true ;
        set(obj.pjtHLinesEpi_, 'Visible', 'on') ;
      end
    end  % function

    function projectionRefreshEPlines_(obj)
      % Update epipolar lines based on working set anchor points.
      % Reads point positions from model.xy_ instead of hPts handles.
      mdl = obj.model_ ;

      if ~mdl.isCalRig
        return ;
      end

      pjtIPtsLocal = mdl.pjtIPts ;
      for i = 1:length(pjtIPtsLocal)
        if isnan(pjtIPtsLocal(i))
          continue ;
        end
        iPt1 = pjtIPtsLocal(i) ;
        xy1 = mdl.xy(iPt1, :) ;
        iAx1 = mdl.iPt2iAx_(iPt1) ;
        clr = obj.hPts_(iPt1).Color ;
        crig = mdl.pjtCalRig_ ;
        % update text marker to include an 'a'
        set(obj.hPtsTxt_(iPt1), 'String', [obj.hPtsTxtStrs_{iPt1} 'a']) ;
        for iAx = 1:mdl.nView
          if iAx == i
            continue ;
          end
          hIm = obj.hIms_(iAx) ;
          imroi = [hIm.XData hIm.YData] ;

          [x, y] = crig.computeEpiPolarLine(iAx1, xy1, iAx, imroi) ;

          hEpi = obj.pjtHLinesEpi_(iAx, i) ;
          set(hEpi, 'XData', x, 'YData', y, ...
            'Visible', onIff(mdl.showCalibration_), 'Color', clr) ;
        end
      end
    end  % function

    function projectionRefreshReconPts_(obj)
      % Update reconstructed points based on two anchor points.
      % Reads point positions from model.xy_ instead of hPts handles.
      mdl = obj.model_ ;

      assert(mdl.pjtState == 2) ;
      if ~mdl.isCalRig
        return ;
      end

      pjtIPtsLocal = mdl.pjtIPts ;
      iPt1 = pjtIPtsLocal(1) ;
      iPt2 = pjtIPtsLocal(2) ;
      iAx1 = mdl.iPt2iAx_(iPt1) ;
      iAx2 = mdl.iPt2iAx_(iPt2) ;

      xy1 = mdl.xy(iPt1, :) ;
      xy2 = mdl.xy(iPt2, :) ;
      clr = obj.hPts_(iPt1).Color ;
      iAxOther = setdiff(1:mdl.nView, [iAx1 iAx2]) ;
      crig = mdl.pjtCalRig_ ;
      for iAx = iAxOther
        [x, y] = crig.reconstruct(iAx1, xy1, iAx2, xy2, iAx) ;
        set(obj.pjtHLinesRecon_(iAx), ...
          'XData', x, 'YData', y, ...
          'Visible', onIff(mdl.showCalibration_), 'Color', clr) ;
      end
    end  % function

    function projectionSet2nd_(obj, iPt2)
      % Set the second projection anchor point.
      mdl = obj.model_ ;

      assert(~isnan(mdl.pjtIPts(1))) ;
      assert(isnan(mdl.pjtIPts(2))) ;
      assert(iPt2 ~= mdl.pjtIPts(1), 'Second projection point must differ from anchor point.') ;
      set(obj.pjtHLinesEpi_, 'Visible', 'off') ;

      set(obj.hPtsTxt_(iPt2), 'String', [obj.hPtsTxtStrs_{iPt2} 'a']) ;

      obj.projectionRefreshReconPts_() ;
    end  % function

    function projectionRefresh_(obj)
      % Refresh projection display (epipolar lines).
      obj.projectionRefreshEPlines_() ;
    end  % function

    function projectionDisp3DPosn_(obj)
      % Calculate/display triangulated 3D coordinates of current working
      % points.
      mdl = obj.model_ ;

      [X, ~, ~] = mdl.projectionTriangulate() ;

      % C+P LabelerGUI (commented out there)
      u0 = X(1) ;
      v0 = X(2) ;
      w0 = X(3) ;
      VIEWDISTFRAC = 5 ;
      obj.projectionClear3DPosn_() ;
      crig = mdl.pjtCalRig_ ;
      nvw = mdl.nView ;
      for iview = 1:nvw
        ax = obj.hAx_(iview) ;

        xl = ax.XLim ;
        yl = ax.YLim ;
        SCALEMIN = 0 ;
        SCALEMAX = 20 ;
        SCALEN = 300 ;
        avViewSz = (diff(xl) + diff(yl)) / 2 ;
        tgtDX = avViewSz / VIEWDISTFRAC * .8 ;
        scales = linspace(SCALEMIN, SCALEMAX, SCALEN) ;

        [x0, y0] = crig.project3d(u0, v0, w0, iview) ;
        for iScale = 1:SCALEN
          s = scales(iScale) ;
          [x1, y1] = crig.project3d(u0 + s, v0, w0, iview) ;
          [x2, y2] = crig.project3d(u0, v0 + s, w0, iview) ;
          [x3, y3] = crig.project3d(u0, v0, w0 + s, iview) ;
          d1 = sqrt((x1 - x0).^2 + (y1 - y0).^2) ;
          d2 = sqrt((x2 - x0).^2 + (y2 - y0).^2) ;
          d3 = sqrt((x3 - x0).^2 + (y3 - y0).^2) ;
          if d1 > tgtDX || d2 > tgtDX || d3 > tgtDX
            fprintf(1, 'Found scale: %.2f\n', s) ;
            break ;
          end
        end

        LINEWIDTH = 5 ;
        FONTSIZE = 22 ;
        lineargs = {'LineWidth', LINEWIDTH} ;
        textargs = {'fontweight', 'bold', 'fontsize', FONTSIZE, 'parent', ax} ;
        obj.pjtShow3D_(end+1, 1) = plot(ax, [x0 x1], [y0 y1], 'r-', lineargs{:}) ;
        obj.pjtShow3D_(end+1, 1) = text(x1, y1, 'x', 'Color', [1 0 0], textargs{:}) ;
        obj.pjtShow3D_(end+1, 1) = plot(ax, [x0 x2], [y0 y2], 'g-', lineargs{:}) ;
        obj.pjtShow3D_(end+1, 1) = text(x2, y2, 'y', 'Color', [0 0.5 0], textargs{:}) ;
        obj.pjtShow3D_(end+1, 1) = plot(ax, [x0 x3], [y0 y3], '-', ...
          'Color', [0 1 1], lineargs{:}) ;
        obj.pjtShow3D_(end+1, 1) = text(x3, y3, 'z', 'Color', [0 0 1], textargs{:}) ;
      end
    end  % function

    function projectionClear3DPosn_(obj)
      % Remove 3D position display handles.
      deleteValidGraphicsHandles(obj.pjtShow3D_) ;
      obj.pjtShow3D_ = gobjects(0, 1) ;
    end  % function

  end  % methods

  %% Working set graphics
  methods

    function projectionWorkingSetClear_(obj)
      % Clear working set graphics: reset all point text to normal.
      mdl = obj.model_ ;
      h = obj.hPtsTxt_ ;
      hClrs = mdl.ptsPlotInfo.Colors ;
      for i = 1:mdl.nPts
        set(h(i), 'Color', hClrs(i, :), 'FontWeight', 'normal', 'EdgeColor', 'none') ;
      end
      set(obj.hPts_, 'HitTest', 'on') ;
      obj.labelerController_.currImHud.updateLblPoint(nan, mdl.nPointSet) ;
    end  % function

    function projectionWorkingSetSet_(obj, iSet)
      % Set working set graphics: bold working set points, dim others.
      mdl = obj.model_ ;
      iPtsSet = mdl.iSet2iPt_(iSet, :) ;

      h = obj.hPts_ ;
      hPT = obj.hPtsTxt_ ;
      hClrs = mdl.ptsPlotInfo.Colors ;
      for i = 1:mdl.nPts
        if any(i == iPtsSet)
          set(hPT(i), 'Color', hClrs(i, :), 'FontWeight', 'bold', 'EdgeColor', 'w') ;
          set(h(i), 'HitTest', 'on') ;
        else
          set(hPT(i), 'Color', hClrs(i, :)*.75, 'FontWeight', 'normal', 'EdgeColor', 'none') ;
          set(h(i), 'HitTest', 'off') ;
        end
      end
      iAx = find(get(0, 'CurrentFigure') == obj.hFig_) ;
      if isscalar(iAx)
        set(obj.hFig_(iAx), 'CurrentObject', get(hPT(iAx), 'Parent')) ;
      end
      obj.labelerController_.currImHud.updateLblPoint(iSet, mdl.nPointSet) ;
      obj.projectionRefresh_() ;
    end  % function

    function refreshHotkeyDesc(obj)
      % Update xlabel and txLblCoreAux strings for hotkey mapping.
      mdl = obj.model_ ;
      iSet0 = mdl.numHotKeyPtSet_ ;
      iSet1 = iSet0 + 9 ;
      str = sprintf('Hotkeys 1-9,0 map to 3d points %d-%d, ` (backquote) toggles', iSet0, iSet1) ;
      [obj.hAxXLabels_(2:end).String] = deal(str) ;
      obj.txLblCoreAux_.String = str ;
    end  % function

  end  % methods

  %% Presentation
  methods

    function shortcuts = LabelShortcuts(obj)
      % Return shortcut descriptions for MultiView Calibrated mode.

      shortcuts = cell(0, 3) ;

      shortcuts{end+1, 1} = 'Toggle whether epipolar lines are shown' ;
      shortcuts{end, 2} = 'space' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Accept current labels' ;
      shortcuts{end, 2} = 's' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'Toggle whether selected kpt is occluded' ;
      shortcuts{end, 2} = 'o' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = 'If fully-occluded box is shown, move selected keypoint to occluded box.' ;
      shortcuts{end, 2} = 'u' ;
      shortcuts{end, 3} = {} ;

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

      DXFAC = LabelCoreMultiViewCalibrated2Model.DXFAC ;
      DXFACBIG = LabelCoreMultiViewCalibrated2Model.DXFACBIG ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move right by 1/%.1fth of axis size, ow forward one frame', DXFAC) ;
      shortcuts{end, 2} = 'Right arrow' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move left by 1/%.1fth of axis size, ow back one frame', DXFAC) ;
      shortcuts{end, 2} = 'Left arrow' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move up by 1/%.1fth of axis size', DXFAC) ;
      shortcuts{end, 2} = 'Up arrow' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move down by 1/%.1f of axis size', DXFAC) ;
      shortcuts{end, 2} = 'Down arrow' ;
      shortcuts{end, 3} = {} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move right by 1/%.1fth of axis size', DXFACBIG) ;
      shortcuts{end, 2} = '+' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move left by 1/%.1fth of axis size', DXFACBIG) ;
      shortcuts{end, 2} = '-' ;
      shortcuts{end, 3} = {'Shift'} ;

      mdl = obj.model_ ;
      shortcuts{end+1, 1} = ...
        sprintf('If kpt selected, move left by 1/%.1fth of axis size ow go to next %s', DXFACBIG, ...
          obj.labeler_.movieShiftArrowNavMode.prettyStr) ;
      shortcuts{end, 2} = 'Left arrow' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = ...
        sprintf('If kpt selected, move right by 1/%.1fth of axis size, ow go to previous %s', DXFACBIG, ...
          obj.labeler_.movieShiftArrowNavMode.prettyStr) ;
      shortcuts{end, 2} = 'Right arrow' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move up by 1/%.1fth of axis size', DXFACBIG) ;
      shortcuts{end, 2} = 'Up arrow' ;
      shortcuts{end, 3} = {'Shift'} ;

      shortcuts{end+1, 1} = sprintf('If kpt selected, move down by 1/%.1fth of axis size', DXFACBIG) ;
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
      % Return labeling help text for MultiView Calibrated mode.
      h = cell(0, 1) ;
      h{end+1} = 'Adjust all keypoints for all views, then click Accept to store. ' ;
      h{end+1} = '' ;
      h{end+1} = ['There is a set of template/"white" points on the ', ...
        'image at all times. To ', ...
        'label a frame, adjust the points as necessary and accept. Adjusted ', ...
        'points are shown in colors (rather than white). '] ;
      h{end+1} = '' ;
      h{end+1} = ['Select a keypoint by typing the number identifying it. ', ...
        'If you have more than 10 keypoints, the ` (backquote) key lets you ', ...
        'change which set of 10 keypoints the number keys correspond to.'] ;
      h{end+1} = ['Once a keypoint is selected, it can be adjusted in any of ', ...
        'the views by clicking on the desired location in an image. ', ...
        'Fine adjustments can be made using the arrow keys. '] ;
      h{end+1} = ['Type the keypoint number again to deselect it, or type another ', ...
        'keypoint number to select a different keypoint. '] ;
      h{end+1} = ['If no keypoints are selected, you can adjust any keypoint by ', ...
        'clicking down on it and dragging it to the desired location.'] ;
      h{end+1} = '' ;
      h{end+1} = ['To convert from 2D keypoint coordinates in each view to a 3D ', ...
        'coordinate, the cameras must be calibrated and calibration information must be ', ...
        'loaded into the project. '] ;
      h{end+1} = ['If calibration information is provided, "epipolar" lines can be shown ', ...
        'to help with labeling. If we know the 2-D coordinate of a point in one view, the ', ...
        'epipolar line is the line that this point may lie on in another view. '] ;
      h{end+1} = ['Epipolar lines will be shown for keypoints after they are adjusted in any ', ...
        'of the ways described above. An "a" will appear next to the keypoint number to ', ...
        'indicate that its epipolar line is being shown. '] ;
      h{end+1} = 'You can toggle whether epipolar lines are shown or hidden with the space key.' ;
      h{end+1} = '' ;
      h1 = obj.getLabelingHelp@LabelCoreController() ;
      h = [h(:) ; h1(:)] ;
    end  % function

  end  % methods

end  % classdef
