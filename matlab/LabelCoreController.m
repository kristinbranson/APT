classdef LabelCoreController < handle
% Labeling controller base class
%
% Owns all graphics handles for labeling: point markers, text labels,
% skeleton edges, occluded box points. Receives GUI callbacks (axis clicks,
% key presses, mouse motion) and delegates data logic to LabelCoreModel.
% Listens to model events and syncs graphics to model state.
%
% This is the controller half of the LabelCore MVC split. The model
% half is LabelCoreModel.

  properties (Abstract)
    supportsSingleView      % scalar logical (mirrors model for convenience)
    supportsMultiView       % scalar logical
    supportsCalibration     % scalar logical
    supportsMultiAnimal     % scalar logical
  end

  properties (Transient)
    labelerController_      % scalar LabelerController
    model_                  % scalar LabelCoreModel

    hFig_                   % [nview] figure handles (first is main fig)
    hAx_                    % [nview] axis handles (first is main axis)
    hIms_                   % [nview] image handles
    hAxOcc_                 % [nview] scalar handle, occluded-axis
    tbAccept_               % scalar handle, togglebutton
    pbClear_                % scalar handle, clearbutton
    txLblCoreAux_           % scalar handle, auxiliary text

    hPts_                   % [nPts x 1] point handles
    hPtsTxt_                % [nPts x 1] text handles
    hPtsOcc_                % [nPts x 1] occluded point handles
    hPtsTxtOcc_             % [nPts x 1] occluded text handles
    hSkel_                  % [nEdges x 1] skeleton handles

    listeners_              % event.listener array, model event subscriptions
  end

  methods (Static)

    function obj = create(labelerController, model, labelMode)
      % Create the LabelCoreController subclass corresponding to labelMode.
      switch labelMode
        case LabelMode.SEQUENTIAL
          obj = LabelCoreSeqController(labelerController, model) ;
        case LabelMode.SEQUENTIALADD
          obj = LabelCoreSeqAddController(labelerController, model) ;
        case LabelMode.TEMPLATE
          obj = LabelCoreTemplateController(labelerController, model) ;
        case LabelMode.HIGHTHROUGHPUT
          obj = LabelCoreHTController(labelerController, model) ;
        case LabelMode.MULTIVIEWCALIBRATED2
          obj = LabelCoreMultiViewCalibrated2Controller(labelerController, model) ;
        case LabelMode.MULTIANIMAL
          obj = LabelCoreSeqMAController(labelerController, model) ;
        otherwise
          error('Unknown label mode %s', char(labelMode)) ;
      end
    end  % function

  end  % methods (Static)

  methods (Sealed=true)

    function obj = LabelCoreController(labelerController, model)
      % Construct a LabelCoreController, storing references.
      obj.labelerController_ = labelerController ;
      obj.model_ = model ;

      gd = labelerController ;
      obj.hFig_ = gd.figs_all ;
      obj.hAx_ = gd.axes_all ;
      obj.hIms_ = gd.images_all ;
      obj.hAxOcc_ = gd.axes_occ ;
      obj.tbAccept_ = gd.tbAccept ;
      obj.pbClear_ = gd.pbClear ;
      obj.txLblCoreAux_ = gd.txLblCoreAux ;
      set(obj.tbAccept_, 'Style', 'togglebutton') ;
    end  % function

    function init(obj)
      % Initialize graphics: create point handles, skeleton, register callbacks.

      mdl = obj.model_ ;
      nPts = mdl.nPts ;
      ptsPlotInfo = mdl.ptsPlotInfo_ ;

      deleteValidGraphicsHandles(obj.hPts_) ;
      deleteValidGraphicsHandles(obj.hPtsOcc_) ;
      deleteValidGraphicsHandles(obj.hPtsTxt_) ;
      deleteValidGraphicsHandles(obj.hPtsTxtOcc_) ;
      deleteValidGraphicsHandles(obj.hSkel_) ;
      obj.hPts_ = gobjects(nPts, 1) ;
      obj.hPtsOcc_ = [] ;
      obj.hPtsTxt_ = gobjects(nPts, 1) ;
      obj.hPtsTxtOcc_ = [] ;

      ax = obj.hAx_ ;
      obj.updateSkeletonEdges() ;

      pvMarker = struct2paramscell(ptsPlotInfo.MarkerProps) ;
      pvText = struct2paramscell(ptsPlotInfo.TextProps) ;

      for i = 1:nPts
        ptsArgs = {nan, nan, pvMarker{:}, ...
          'Color', ptsPlotInfo.Colors(i,:), ...
          'UserData', i} ; %#ok<CCAT>
        obj.hPts_(i) = plot(ax(1), ptsArgs{:}, ...
          'Tag', sprintf('LabelCore_Pts_%d', i)) ;
        obj.hPtsTxt_(i) = text(nan, nan, num2str(i), 'Parent', ax(1), ...
          pvText{:}, ...
          'Color', ptsPlotInfo.Colors(i,:), ...
          'PickableParts', 'none', ...
          'Tag', sprintf('LabelCore_Pts_%d', i)) ;
      end

      set(obj.hAx_, 'ButtonDownFcn', @(s,e)obj.axBDF(s,e)) ;
      arrayfun(@(x)set(x, 'HitTest', 'on', 'ButtonDownFcn', @(s,e)obj.ptBDF(s,e)), obj.hPts_) ;
      gdata = obj.labelerController_ ;
      set(gdata.uipanel_curr, 'ButtonDownFcn', @(s,e)obj.pnlBDF(s,e)) ;

      set(gdata.tbAccept, 'Enable', 'on') ;
      set(gdata.pbClear, 'Enable', 'on') ;
      mdl.labeler_.currImHud.updateReadoutFields('hasLblPt', false) ;

      if mdl.labeler_.showOccludedBox
        obj.showOcc() ;
      end

      obj.txLblCoreAux_.Visible = 'off' ;
      units0 = obj.txLblCoreAux_.FontUnits ;
      obj.txLblCoreAux_.FontUnits = 'pixels' ;
      obj.txLblCoreAux_.FontSize = 12 ;
      obj.txLblCoreAux_.FontUnits = units0 ;

      obj.updateShowSkeleton() ;

      % Register model listeners
      obj.listeners_ = [ ...
        addlistener(mdl, 'update',             @(s,e)obj.onUpdate()) ; ...
        addlistener(mdl, 'updateLabelCoords',  @(s,e)obj.onUpdateLabelCoords()) ; ...
        addlistener(mdl, 'updateLabelCoordsI', @(s,e)obj.onUpdateLabelCoordsI()) ; ...
        addlistener(mdl, 'updateState',        @(s,e)obj.onUpdateState()) ; ...
        addlistener(mdl, 'updateOccluded',     @(s,e)obj.onUpdateOccluded()) ; ...
        addlistener(mdl, 'updateEstOccluded',  @(s,e)obj.onUpdateEstOccluded()) ; ...
        addlistener(mdl, 'updateSelected',     @(s,e)obj.onUpdateSelected()) ; ...
        addlistener(mdl, 'updateHideLabels',   @(s,e)obj.onUpdateHideLabels()) ; ...
      ] ;

      obj.initHook() ;
    end  % function

  end  % methods (Sealed=true)

  methods

    function delete(obj)
      % Clean up graphics handles.
      deleteValidGraphicsHandles(obj.hPts_) ;
      deleteValidGraphicsHandles(obj.hPtsTxt_) ;
      deleteValidGraphicsHandles(obj.hPtsOcc_) ;
      deleteValidGraphicsHandles(obj.hPtsTxtOcc_) ;
      deleteValidGraphicsHandles(obj.hSkel_) ;
      delete(obj.listeners_) ;
    end  % function

    function initHook(obj) %#ok<MANU>
      % Called from init(). Override in subclasses.
    end  % function

  end  % methods

  %% Model event handlers
  methods

    function onUpdate(obj) %#ok<MANU>
      % Full sync fallback. Override in subclasses.
    end  % function

    function onUpdateLabelCoords(obj)
      % Sync all point graphics to model xy_.
      obj.syncAllPointGraphics() ;
    end  % function

    function onUpdateLabelCoordsI(obj)
      % Sync single point graphics to model xy_ for lastChangedIPt_.
      iPt = obj.model_.lastChangedIPt_ ;
      obj.syncPointGraphicsI(iPt) ;
    end  % function

    function onUpdateState(obj) %#ok<MANU>
      % State changed. Override in subclasses for state-dependent UI.
    end  % function

    function onUpdateOccluded(obj)
      % Occluded flags changed. Refresh occluded point display.
      obj.refreshOccludedPts() ;
    end  % function

    function onUpdateEstOccluded(obj)
      % Estimated-occluded flags changed. Refresh markers.
      obj.refreshPtMarkers() ;
    end  % function

    function onUpdateSelected(obj)
      % Selection changed. Refresh markers.
      obj.refreshPtMarkers() ;
    end  % function

    function onUpdateHideLabels(obj)
      % Hide/show labels state changed.
      if obj.model_.hideLabels_
        obj.labelsHide() ;
      else
        obj.labelsShow() ;
      end
    end  % function

  end  % methods

  %% GUI callback handlers
  methods

    function axBDF(obj, src, evt) %#ok<INUSD>
      % Axis button-down handler. Override in subclasses.
    end  % function

    function ptBDF(obj, src, evt) %#ok<INUSD>
      % Point button-down handler. Override in subclasses.
    end  % function

    function pnlBDF(obj, src, evt)
      % Panel button-down handler: forward clicks within axis bounds to axBDF.
      if ~obj.model_.labeler_.isReady
        return ;
      end
      tmp = get(obj.hAx_(1), 'CurrentPoint') ;
      pos = tmp(1, 1:2) ;
      xlim = get(obj.hAx_(1), 'XLim') ;
      ylim = get(obj.hAx_(1), 'YLim') ;
      if pos(1) >= xlim(1) && pos(1) <= xlim(2) && pos(2) >= ylim(1) && pos(2) <= ylim(2)
        evtForAxes = struct('Button', evt.Button, ...
                            'IntersectionPoint', [pos 0]) ;
        obj.axBDF(src, evtForAxes) ;
      end
    end  % function

    function axOccBDF(obj, src, evt) %#ok<INUSD>
      % Occluded-axis button-down handler. Override in subclasses.
    end  % function

    function wbmf(obj, src, evt) %#ok<INUSD>
      % Window button motion handler. Override in subclasses.
    end  % function

    function wbuf(obj, src, evt) %#ok<INUSD>
      % Window button up handler. Override in subclasses.
    end  % function

    function tfKPused = kpf(obj, src, evt) %#ok<INUSD>
      % Key press handler. Override in subclasses.
      tfKPused = false ;
    end  % function

    function v = isPanZoom(obj, figi)
      % Check whether the user is in pan-zoom mode (modifier key held).
      if ~exist('figi', 'var')
        figi = 1 ;
      end
      if ishandle(obj.hFig_(figi)) && ...
          ismember(obj.model_.panZoomMod_, obj.hFig_(figi).CurrentModifier)
        v = true ;
      else
        v = false ;
      end
    end  % function

  end  % methods

  %% Occluded box
  methods (Sealed=true)

    function showOcc(obj)
      % Create occluded-box point handles and register callbacks.

      mdl = obj.model_ ;
      nPts = mdl.nPts ;
      ptsPlotInfo = mdl.ptsPlotInfo_ ;

      deleteValidGraphicsHandles(obj.hPtsOcc_) ;
      deleteValidGraphicsHandles(obj.hPtsTxtOcc_) ;
      obj.hPtsOcc_ = gobjects(nPts, 1) ;
      obj.hPtsTxtOcc_ = gobjects(nPts, 1) ;

      axOcc = obj.hAxOcc_ ;
      pvMarker = struct2paramscell(ptsPlotInfo.MarkerProps) ;
      pvText = struct2paramscell(ptsPlotInfo.TextProps) ;

      for i = 1:nPts
        ptsArgs = {nan, nan, pvMarker{:}, ...
          'Color', ptsPlotInfo.Colors(i,:), ...
          'UserData', i} ; %#ok<CCAT>
        obj.hPtsOcc_(i) = plot(axOcc(1), ptsArgs{:}, ...
          'Tag', sprintf('LabelCore_PtsOcc_%d', i)) ;
        obj.hPtsTxtOcc_(i) = text(nan, nan, num2str(i), 'Parent', axOcc(1), ...
          pvText{:}, ...
          'Color', ptsPlotInfo.Colors(i,:), ...
          'PickableParts', 'none', ...
          'Tag', sprintf('LabelCore_Pts_%d', i)) ;
      end

      arrayfun(@(x)set(x, 'HitTest', 'on', 'ButtonDownFcn', @(s,e)obj.ptBDF(s,e)), ...
        obj.hPtsOcc_) ;
      set(obj.hAxOcc_, 'ButtonDownFcn', @(s,e)obj.axOccBDF(s,e)) ;

      obj.showOccHook() ;
      obj.refreshOccludedPts() ;
    end  % function

    function hideOcc(obj)
      % Remove occluded-box point handles.
      deleteValidGraphicsHandles(obj.hPtsOcc_) ;
      deleteValidGraphicsHandles(obj.hPtsTxtOcc_) ;
      obj.hPtsOcc_ = [] ;
      obj.hPtsTxtOcc_ = [] ;
      set(obj.hAxOcc_, 'ButtonDownFcn', '') ;
    end  % function

  end  % methods (Sealed=true)

  methods

    function showOccHook(obj) %#ok<MANU>
      % Override in subclasses for additional occ-box setup.
    end  % function

  end  % methods

  %% Graphics sync
  methods

    function syncAllPointGraphics(obj)
      % Sync all point handles to model coordinates. Handles occlusion.

      mdl = obj.model_ ;
      xy = mdl.xy_ ;
      ptsPlotInfo = mdl.ptsPlotInfo_ ;
      tfOccld = mdl.tfOcc_ ;

      obj.setPtsCoords(xy(~tfOccld, :), obj.hPts_(~tfOccld), obj.hPtsTxt_(~tfOccld)) ;

      % Skeleton
      LabelCoreController.setSkelCoords(xy, tfOccld, obj.hSkel_, mdl.skeletonEdges()) ;

      % Occluded handling
      obj.refreshOccludedPts() ;

      % Tags / estimated occluded
      if any(mdl.tfEstOcc_)
        obj.refreshPtMarkers() ;
      end
    end  % function

    function syncPointGraphicsI(obj, iPt)
      % Sync point handle(s) for specific point index to model coordinates.
      mdl = obj.model_ ;
      xy = mdl.xy_(iPt, :) ;
      obj.setPtsCoords(xy, obj.hPts_(iPt), obj.hPtsTxt_(iPt)) ;

      % Update skeleton edges connected to this point
      edges = mdl.skeletonEdges() ;
      for i = 1:numel(iPt)
        [js, ks] = find(edges == iPt(i)) ;
        for jj = 1:numel(js)
          j = js(jj) ;
          k = ks(jj) ;
          xdata = get(obj.hSkel_(j), 'XData') ;
          ydata = get(obj.hSkel_(j), 'YData') ;
          xdata(k) = xy(i, 1) ;
          ydata(k) = xy(i, 2) ;
          set(obj.hSkel_(j), 'XData', xdata, 'YData', ydata) ;
        end
      end
    end  % function

    function refreshOccludedPts(obj)
      % Set point/text locs based on model.tfOcc_.
      mdl = obj.model_ ;
      tf = mdl.tfOcc_ ;
      nOcc = nnz(tf) ;

      if nOcc > 0
        if isempty(obj.hPtsOcc_)
          mdl.labeler_.setShowOccludedBox(true) ;
          obj.showOcc() ;
          return ;
        end
        % Hide the 'regular' pts that are fully-occ
        obj.setPtsCoords(nan(nOcc, 2), obj.hPts_(tf), obj.hPtsTxt_(tf)) ;
      end

      if ~isempty(obj.hPtsOcc_)
        iOcc = find(tf) ;
        setPositionsOfLabelLinesAndTextsBangBang( ...
          obj.hPtsOcc_(tf), obj.hPtsTxtOcc_(tf), [iOcc(:) ones(nOcc, 1)], 0.25) ;
        setPositionsOfLabelLinesAndTextsBangBang( ...
          obj.hPtsOcc_(~tf), obj.hPtsTxtOcc_(~tf), nan(mdl.nPts - nOcc, 2), 0.25) ;
      end
    end  % function

    function refreshPtMarkers(obj, varargin)
      % Update point markers based on model tfEstOcc_ and tfSel_.
      mdl = obj.model_ ;
      [iPts, doPtsOcc] = myparse(varargin, ...
        'iPts', 1:mdl.nPts, ...
        'doPtsOcc', false) ;

      ppi = mdl.ptsPlotInfo_ ;
      ppitm = ppi.TemplateMode ;

      hPoints = obj.hPts_(iPts) ;
      tfSl = mdl.tfSel_(iPts) ;
      tfEO = mdl.tfEstOcc_(iPts) ;

      set(hPoints(tfSl & tfEO), 'Marker', ppitm.SelectedOccludedMarker) ;
      set(hPoints(tfSl & ~tfEO), 'Marker', ppitm.SelectedPointMarker) ;
      set(hPoints(~tfSl & tfEO), 'Marker', ppi.OccludedMarker) ;
      set(hPoints(~tfSl & ~tfEO), 'Marker', ppi.MarkerProps.Marker) ;

      if doPtsOcc && ~isempty(obj.hPtsOcc_)
        hPointsOcc = obj.hPtsOcc_(iPts) ;
        set(hPointsOcc(tfSl), 'Marker', ppitm.SelectedPointMarker) ;
        set(hPointsOcc(~tfSl), 'Marker', ppi.MarkerProps.Marker) ;
      end
    end  % function

    function refreshTxLabelCoreAux(obj)
      % Update the auxiliary text label showing hotkey mapping.
      iPt0 = obj.model_.kpfIPtFor1Key_ ;
      iPt1 = iPt0 + 9 ;
      str = sprintf('Hotkeys 1-9,0 map to points %d-%d, ` (backquote) toggles', iPt0, iPt1) ;
      obj.txLblCoreAux_.String = str ;
    end  % function

  end  % methods

  %% Show/hide viz
  methods

    function labelsHide(obj)
      % Hide all label point graphics.
      [obj.hPts_.Visible] = deal('off') ;
      [obj.hPtsTxt_.Visible] = deal('off') ;
      obj.updateShowSkeleton() ;
    end  % function

    function labelsShow(obj)
      % Show all label point graphics.
      [obj.hPts_.Visible] = deal('on') ;
      [obj.hPtsTxt_.Visible] = deal('on') ;
      obj.updateShowSkeleton() ;
    end  % function

    function labelsHideToggle(obj)
      % Toggle label visibility via the model.
      mdl = obj.model_ ;
      mdl.setHideLabels(~mdl.hideLabels_) ;
    end  % function

    function updateShowSkeleton(obj)
      % Show or hide skeleton edges.
      if isempty(obj.hSkel_)
        return ;
      end
      if obj.model_.labeler_.showSkeleton && ~obj.model_.hideLabels_
        [obj.hSkel_.Visible] = deal('on') ;
      else
        [obj.hSkel_.Visible] = deal('off') ;
      end
    end  % function

  end  % methods

  %% Cosmetics
  methods

    function updateColors(obj, colors)
      % Update colors for point markers and text labels.
      mdl = obj.model_ ;
      mdl.ptsPlotInfo_.Colors = colors ;

      for i = 1:mdl.nPts
        clrI = colors(i, :) ;
        if numel(obj.hPts_) >= i && ishandle(obj.hPts_(i))
          set(obj.hPts_(i), 'Color', clrI) ;
        end
        if numel(obj.hPtsOcc_) >= i && ishandle(obj.hPtsOcc_(i))
          set(obj.hPtsOcc_(i), 'Color', clrI) ;
        end
        if numel(obj.hPtsTxt_) >= i && ishandle(obj.hPtsTxt_(i))
          set(obj.hPtsTxt_(i), 'Color', clrI) ;
        end
        if numel(obj.hPtsTxtOcc_) >= i && ishandle(obj.hPtsTxtOcc_(i))
          set(obj.hPtsTxtOcc_(i), 'Color', clrI) ;
        end
      end
    end  % function

    function updateMarkerCosmetics(obj, pvMarker)
      % Update marker cosmetics for all point handles.
      mdl = obj.model_ ;
      flds = fieldnames(pvMarker) ;
      for f = flds(:)' , f = f{1} ; %#ok<FXSET>
        mdl.ptsPlotInfo_.MarkerProps.(f) = pvMarker.(f) ;
      end
      set(obj.hPts_, pvMarker) ;
      set(obj.hPtsOcc_, pvMarker) ;
    end  % function

    function updateTextLabelCosmetics(obj, pvText, txtoffset)
      % Update text label cosmetics.
      mdl = obj.model_ ;
      flds = fieldnames(pvText) ;
      for f = flds(:)' , f = f{1} ; %#ok<FXSET>
        mdl.ptsPlotInfo_.TextProps.(f) = pvText.(f) ;
      end
      set(obj.hPtsTxt_, pvText) ;
      mdl.ptsPlotInfo_.TextOffset = txtoffset ;
      obj.redrawTextLabels() ;
    end  % function

    function skeletonCosmeticsUpdated(obj)
      % Refresh skeleton edge cosmetics from labeler.
      lObj = obj.model_.labeler_ ;
      ppi = lObj.labelPointsPlotInfo ;
      obj.model_.ptsPlotInfo_.SkeletonProps = ppi.SkeletonProps ;
      set(obj.hSkel_, ppi.SkeletonProps) ;
    end  % function

    function updateSkeletonEdges(obj)
      % Rebuild skeleton edge handles.
      mdl = obj.model_ ;
      ax = obj.hAx_ ;
      ptsPlotInfo = mdl.ptsPlotInfo_ ;
      edges = mdl.skeletonEdges() ;

      deleteValidGraphicsHandles(obj.hSkel_) ;
      obj.hSkel_ = gobjects(size(edges, 1), 1) ;
      for i = 1:size(edges, 1)
        obj.hSkel_(i) = LabelCoreController.initSkeletonEdge(ax, i, ptsPlotInfo) ;
      end
      xy = mdl.getLabelCoords() ;
      tfOccld = any(isinf(xy), 2) ;
      LabelCoreController.setSkelCoords(xy, tfOccld, obj.hSkel_, edges) ;
    end  % function

  end  % methods

  %% Low-level graphics helpers
  methods

    function setPtsCoords(obj, xy, hPts, hTxt) %#ok<INUSL>
      % Set coordinates for point and text handles.
      txtOffset = obj.model_.labeler_.labelPointsPlotInfo.TextOffset ;
      setPositionsOfLabelLinesAndTextsBangBang(hPts, hTxt, xy, txtOffset) ;
    end  % function

    function redrawTextLabels(obj)
      % Redraw text labels (e.g. when text offset is updated).
      txtOffset = obj.model_.labeler_.labelPointsPlotInfo.TextOffset ;
      h = obj.hPts_ ;
      hT = obj.hPtsTxt_ ;
      x = get(h, 'XData') ;
      y = get(h, 'YData') ;
      xy = [cell2mat(x(:)) cell2mat(y(:))] ;
      xyT = xy + txtOffset ;
      for i = 1:numel(hT)
        set(hT(i), 'Position', [xyT(i, 1) xyT(i, 2) 1]) ;
      end
    end  % function

  end  % methods

  %% Presentation
  methods

    function shortcuts = LabelShortcuts(obj) %#ok<MANU>
      % Return shortcut descriptions. Override in subclasses.
      shortcuts = cell(0, 3) ;
    end  % function

    function h = getLabelingHelp(obj)
      % Return labeling help text. Override in subclasses.
      h = cell(0, 1) ;
      h{end+1} = '{\bf{Shortcuts}}:' ;
      shortcuts = obj.LabelShortcuts() ;
      for i = 1:size(shortcuts, 1)
        desc = shortcuts{i, 1} ;
        key = shortcuts{i, 2} ;
        mod = shortcuts{i, 3} ;
        if ~isempty(mod)
          key = [sprintf('%s ', mod{:}), key] ; %#ok<AGROW>
        end
        h{end+1} = sprintf('{\\fontname{Courier} %s }: %s.', key, desc) ; %#ok<AGROW>
      end
    end  % function

  end  % methods

  %% Static graphics utilities
  methods (Static)

    function xy = getCoordsFromPts(hPts)
      % Extract XY coordinates from point handles.
      x = get(hPts, {'XData'}) ;
      y = get(hPts, {'YData'}) ;
      x = cell2mat(x) ;
      y = cell2mat(y) ;
      xy = [x y] ;
    end  % function

    function setPtsColor(hPts, hTxt, colors)
      % Set colors on point and text handles.
      assert(numel(hPts) == numel(hTxt)) ;
      n = numel(hPts) ;
      assert(isequal(size(colors), [n 3])) ;
      for i = 1:n
        clr = colors(i, :) ;
        set(hPts(i), 'Color', clr) ;
        set(hTxt(i), 'Color', clr) ;
      end
    end  % function

    function h = initSkeletonEdge(ax, i, ptsPlotInfo)
      % Create a single skeleton edge line handle.
      skelprops = ptsPlotInfo.SkeletonProps ;
      skelprops = struct2paramscell(skelprops) ;
      h = plot(ax(1), nan(2, 1), nan(2, 1), '-', ...
        'PickableParts', 'none', 'Tag', sprintf('LabelCore_Skel_%d', i), ...
        skelprops{:}) ;
    end  % function

    function setSkelCoords(xy, tfOccld, hSkel, edges)
      % Set skeleton edge coordinates.
      xynan = xy ;
      xynan(tfOccld, :) = nan ;
      for i = 1:numel(hSkel)
        edge = edges(i, :) ;
        set(hSkel(i), 'XData', xynan(edge, 1), 'YData', xynan(edge, 2)) ;
      end
    end  % function

  end  % methods (Static)

end  % classdef
