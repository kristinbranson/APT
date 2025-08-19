classdef LabelCoreMultiViewCalibrated2 < LabelCore
% Multiview labeling for calibrated cameras  

  % Hitting a number key gets you "working on" a certain ptset. All pts
  % in this set bolded; all other pts dimmed. Hitting the same number key
  % un-selects the ptset (which also unselects any pts). Hitting a
  % different number key un-selects the previous ptset and selects a new.
  %
  % Clicking on an axes when a ptset is working will i) jump the working pt
  % in that view to the clicked loc and ii) "select" that pt, turning it into 
  % an 'x' and enabling arrowkeys for fine adjustment. 
  % 
  % All points that are selected and have been adjusted are "anchor" points
  % in previous terminology, and will show epipolar lines and include an
  % 'a' in their label.
  % 
  % Giving a window the focus when a workingset is selected will Select the
  % WSpt in that view.
  %
  % Any WSpt can be dragged.
  %
  % When done, hit another number key to change the working point.
  %
  % When done with all points, hit Accept to Accept labels.
  %
  % This requires a 'CalibratedRig' that knows how to compute epipolar lines
  % and reconstruct 3dpts.
 
  % Alternative description
  %
  % WSset
  % This is the current working set of pts, one in each view. Pts in the
  % current WSset are bolded and other pts are dimmed. Use number hotkeys
  % to select a WSset.
  %
  % SelPt
  % This is the currently selected pt. Only pts in the current WSset can be
  % selected. When selected, a pt is shown as 'x' and can be adjusted with
  % the hotkeys. It can also be anchored/de-anchored with <spacebar>.
  %
  % When a view is given the focus and in particular when its axes is 
  % clicked, the WSset in that view (if any) is automatically selected.
  %
  % Pt adjustment
  % WSset pts can be adjusted by clicking on the various axes. When an axis
  % is clicked, i) the WSpt in that view is jumped to the clicked location,
  % ii) the WSpt in that view is anchored (displacing other anchorpts if 
  % nec), and iii) the pt is selected. Then, fine-adjustments can
  % be done with the arrow keys, since the pt is selected.
  %
  % Alternatively one can simply give the desired view the focus, which
  % will select the WSpt in that view, enabling adjustment with the arrow
  % keys.
  %
  % AnchorSet
  % Epipolar lines for the anchor set are shown in all other views. 
  % 
  % The anchorset starts empty when a WSset first becomes active. All
  % adjusted or accepted and selected points are anchor points. 
 
  properties (SetObservable)
    % If true, streamlined labeling process; labels not shown unless they
    % exist
    streamlined = false; 
  end
  properties
    supportsSingleView = false;
    supportsMultiView = true;
    supportsCalibration = true;
    supportsMultiAnimal = false;
  end
  
  properties
    iPt2iAx       % [npts]. iPt2iAx(iPt) gives the axis index for iPt
    % .hPts       % [npts] from LabelCore. hLine handles for pts (in
    %               respective axes)
    % .hPtsTxt    % [npts]
    %hPtsColors    % [nptsx3] colors for pts, based on prefs.ColorsSets
    hPtsTxtStrs   % [npts] cellstr, text labels for each pt
    iSet2iPt      % [nset x nview]. A point 'set' is a nview-tuple of point 
                  % indices that represent a single physical (3d) point.
                  % .iSet2iPt(iSet,:) are the nview pt indices that
                  % correspond to pointset iSet.
    iPt2iSet      % [npts]. set index for each point.
    showCalibration = true; % whether to show calibration-based info
  end

  properties
    unsupportedKPFFns = {} ;  % cell array of field names for objects that have general keypressfcn 
                              % callbacks but are not supported for this LabelCore
  end
  
  properties (Dependent)
    nView         % scalar
    nPointSet     % scalar, number of 'point sets'.    
    isCalRig
  end  
    
  %% Projections
  properties
    iSetWorking      % scalar. Set index of working set. Can be nan for no working set.

    pjtIPts          % [nview]. NaN if anchor point not clicked for a view.
    pjtHLinesEpi     % [nview * nview]. line handles for epipolar lines. 
                     %    .pjtHLinesEpi(ivw1,ivw2) is EP line shown in ivw1
                     %    originating from ivw2
    pjtHLinesRecon   % [nview]. line handles for reconstructed pts
    
    pjtCalRig         % Scalar some-kind-of-calibration object
    
    pjtShow3D        % [nShow3D] aux handles for showing 3D info
  end
  properties (Dependent)
    pjtState         % either 0, 1, or 2 for number of defined working pts
  end
  
  %% Misc
  properties
    % click-drag
    iPtMove;     % scalar. Either nan, or index of pt being moved
    tfMoved;     % scalar logical; if true, pt being moved was actually moved

    currFrameAdjust; % which frame was most recently adjusted
    tfAdjusted;  % nPts x 1 logical vec. If true, pt has been adjusted from template
    tfStored;  % nPts x 1 logical vec. If true, pt has accepted and stored in labeler object
    
    numHotKeyPtSet; % scalar positive integer. This is the pointset that 
                    % the '1' hotkey currently maps to
    
    hAxXLabels; % [nview] xlabels for axes
    hAxXLabelsFontSize = 11;
    showEpiLines = true;

  end
  
  methods % dep prop getters
    function v = get.nView(obj)
      v = obj.labeler.nview;
    end
    function v = get.nPointSet(obj)
      v = size(obj.iSet2iPt,1);
    end
    function v = get.pjtIPts(obj)
      iSet = obj.iSetWorking;
      v = nan(1,obj.nView);
      if isempty(iSet) || isnan(iSet),
        return;
      end
      for iAx = 1:obj.nView,
        iPt = find(obj.iPt2iSet==iSet & obj.iPt2iAx==iAx);
        if obj.tfAdjusted(iPt),
          v(iAx) = iPt;
        end
      end
    end
    function v = get.pjtState(obj)
      v = nnz(~isnan(obj.pjtIPts));
    end
    function v = get.isCalRig(obj)
      v = ~isempty(obj.pjtCalRig);
    end
  end
  
  methods
    
    function set.numHotKeyPtSet(obj,val)
      obj.numHotKeyPtSet = val;
      obj.refreshHotkeyDesc();
    end

    function setShowCalibration(obj,val)
      obj.showCalibration = val;      
      
      if obj.isCalRig,
        obj.projectionRefresh();
      end

      
    end

    
  end
  
  methods
    
    function obj = LabelCoreMultiViewCalibrated2(varargin)
      obj = obj@LabelCore(varargin{:});
    end
		
    function delete(obj)
      deleteValidGraphicsHandles(obj.pjtHLinesEpi);
      obj.pjtHLinesEpi = [];
      deleteValidGraphicsHandles(obj.pjtHLinesRecon);
      obj.pjtHLinesRecon = [];
      deleteValidGraphicsHandles(obj.hAxXLabels);
      obj.hAxXLabels = [];
    end
        
    function initHook(obj)
      obj.iPt2iAx = obj.labeler.labeledposIPt2View;
      obj.iPt2iSet = obj.labeler.labeledposIPt2Set;
      obj.iSet2iPt = obj.labeler.labeledposIPtSetMap;
      
      % redefine .hPts, .hPtsTxt (originally initted in LabelCore.init())
      % KB 20161213: Added redefining of labeledpos2_ptsH and
      % labeledpos2_ptsTxtH, originally initted in
      % Labelers.labels2VizInit()
      % AL 20170330 At this pt may be able to do multi-view in LabelCore; 
      % alternatively this is acting as multiview lblCore baseclass
      deleteValidGraphicsHandles(obj.hPts);
      deleteValidGraphicsHandles(obj.hPtsTxt);
%       deleteValidGraphicsHandles(obj.labeler.labeledpos2_ptsH);
%       deleteValidGraphicsHandles(obj.labeler.labeledpos2_ptsTxtH);
      deleteValidGraphicsHandles(obj.hSkel);
      obj.hPts = gobjects(obj.nPts,1);
      obj.hPtsTxt = gobjects(obj.nPts,1);
      obj.hSkel = gobjects(size(obj.skeletonEdges,1),1);
%       obj.labeler.labeledpos2_ptsH = gobjects(obj.nPts,1);
%       obj.labeler.labeledpos2_ptsTxtH = gobjects(obj.nPts,1);
      ppi = obj.ptsPlotInfo;
      %obj.hPtsColors = nan(obj.nPointSet,3);
      obj.hPtsTxtStrs = cell(obj.nPts,1);
      
%       ppi2 = obj.labeler.predPointsPlotInfo;
      %ppi2.FontSize = ppi.FontSize;
      
      obj.updateSkeletonEdges();
      obj.updateShowSkeleton();

      pvMarker = struct2paramscell(ppi.MarkerProps);
      pvText = struct2paramscell(ppi.TextProps);
      
      for iPt=1:obj.nPts
        iSet = obj.iPt2iSet(iPt);
        setClr = ppi.Colors(iSet,:);
%         setClr2 = setClr;
        %obj.hPtsColors(iPt,:) = setClr;
        ptsArgs = {nan,nan,pvMarker{:},...
          'ZData',1,... % AL 20160628: seems to help with making points clickable but who knows 2018018 probably remove me after Pickable Parts update
          'Color',setClr,...
          'UserData',iPt,...
          'HitTest','on',...
          'ButtonDownFcn',@(s,e)obj.ptBDF(s,e)}; %#ok<CCAT>
%         ptsArgs2 = {nan,nan,ppi2.Marker,... % AL 2018018: Cant remember why LCMVC2 is messing with labeler.labeledpos2_ptsH
%           'MarkerSize',ppi2.MarkerSize,...  % AL 20190602: Indeed very strange
%           'LineWidth',ppi2.LineWidth,...
%           'Color',setClr2,...
%           'PickableParts','none'};
        ax = obj.hAx(obj.iPt2iAx(iPt));
        obj.hPts(iPt) = plot(ax,ptsArgs{:},'Tag',sprintf('LabelCoreMV_Pt%d',iPt));
%         obj.labeler.labeledpos2_ptsH(iPt) = plot(ax,ptsArgs2{:},...
%           'Tag',sprintf('LabelCoreMV_LabeledPos2%d',iPt)); ...
%           % AL 2018018: Cant remember why LCMVC2 is messing with labeler.labeledpos2_ptsH
        txtStr = num2str(iSet);
        txtargs = {'Color',setClr,pvText{:},'PickableParts','none'}; %#ok<CCAT>
        obj.hPtsTxt(iPt) = text(nan,nan,txtStr,...
          'Parent',ax,txtargs{:},'Tag',sprintf('LabelCoreMV_PtTxt%d',iPt));
        obj.hPtsTxtStrs{iPt} = txtStr;
%         obj.labeler.labeledpos2_ptsTxtH(iPt) = text(nan,nan,txtStr,... % AL 2018018: Cant remember why LCMVC2 is messing with labeler.labeledpos2_ptsH
%           'Parent',ax,...
%           'Color',setClr2,...
%           'FontSize',ppi2.FontSize,...
%           'PickableParts','none',...
%           'Tag',sprintf('LabelCoreMV_LabeledPos2Txt%d',iPt));
      end
      
      obj.setRandomTemplate();
           
      obj.tfAdjusted = false(obj.nPts,1);
      obj.tfStored = false(obj.nPts,1);
      obj.currFrameAdjust = nan;

      obj.hAxXLabels = gobjects(obj.nView,1);
      for iView=1:obj.nView
        ax = obj.hAx(iView);
        obj.hAxXLabels(iView) = xlabel(ax,'','fontsize',obj.hAxXLabelsFontSize);
      end
      obj.txLblCoreAux.Visible = 'on';
      obj.numHotKeyPtSet = 1;
      obj.refreshHotkeyDesc();

      obj.labeler.currImHud.updateReadoutFields('hasLblPt',true);

      obj.projectionWorkingSetClear();
      obj.projectionInit();      
    end
    
    function showOccHook(obj)
      
      deleteValidGraphicsHandles(obj.hPtsOcc);
      deleteValidGraphicsHandles(obj.hPtsTxtOcc);
      obj.hPtsOcc = gobjects(obj.nPts,1);
      obj.hPtsTxtOcc = gobjects(obj.nPts,1);

      ppi = obj.ptsPlotInfo;
      pvMarker = struct2paramscell(ppi.MarkerProps);
      pvText = struct2paramscell(ppi.TextProps);

      for iPt=1:obj.nPts
        iSet = obj.iPt2iSet(iPt);
        setClr = ppi.Colors(iSet,:);
        %obj.hPtsColors(iPt,:) = setClr;
        ptsArgsOcc = {nan,nan,pvMarker{:},...
          'Color',setClr,...
          'UserData',iPt,...
          'HitTest','off'}; %#ok<CCAT>
        axocc = obj.hAxOcc(obj.iPt2iAx(iPt));
        obj.hPtsOcc(iPt) = plot(axocc,ptsArgsOcc{:},'Tag',sprintf('LabelCoreMV_PtOcc%d',iPt));
        txtStr = num2str(iSet);
        txtargs = {'Color',setClr,...
          pvText{:},...
          'PickableParts','none'}; %#ok<CCAT>
        obj.hPtsTxtOcc(iPt) = text(nan,nan,txtStr,...
          'Parent',axocc,txtargs{:},'Tag',sprintf('LabelCoreMV_PtTxtOcc%d',iPt));
      end
      
      for iVw=1:obj.nView
        axis(obj.hAxOcc(iVw),[0 obj.nPointSet+1 0 2]);
      end
      
    end
    
    function edges = skeletonEdges(obj)
      
      se = obj.labeler.skeletonEdges;
      nEdges = size(se,1);
      edges = repmat(se,[obj.nView,1]);
      for ivw = 1:obj.nView,
        edges((ivw-1)*nEdges+1:ivw*nEdges,:) = reshape(obj.iSet2iPt(se(:),ivw),[nEdges,2]);
      end
      
    end
    
    function updateSkeletonEdges(obj)
      
      if isempty(obj.iSet2iPt) || isempty(obj.labeler.skeletonEdges),
        return;
      end
      
      ax = obj.hAx;
      ptsPlotInfo = obj.ptsPlotInfo;
      
      deleteValidGraphicsHandles(obj.hSkel);
      nEdgesPerView = size(obj.skeletonEdges,1)/obj.nView;
      obj.hSkel = gobjects(size(obj.skeletonEdges,1),1);
      for ivw = 1:obj.nView,
        for i = 1:size(obj.labeler.skeletonEdges,1),
          iEdge = (ivw-1)*nEdgesPerView+i;
          obj.hSkel(iEdge) = LabelCore.initSkeletonEdge(ax(ivw),iEdge,ptsPlotInfo);
        end
      end
      xy = obj.getLabelCoords();
      tfOccld = any(isinf(xy),2);
      LabelCore.setSkelCoords(xy,tfOccld,obj.hSkel,obj.skeletonEdges);      
    end
    
  end
  
  methods

    % newFrameAndTarget() combines all the brains of transitions for 
    % convenience reasons
    
    function newFrame(obj,iFrm0,iFrm1,iTgt,tfForceUpdate)
      if nargin < 5
        tfForceUpdate = false;
      end
      obj.newFrameAndTarget(iFrm0,iFrm1,iTgt,iTgt,tfForceUpdate);
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm)
      obj.newFrameAndTarget(iFrm,iFrm,iTgt0,iTgt1);
    end
    
    function newFrameAndTarget(obj,iFrm0,iFrm1,iTgt0,iTgt1,tfForceUpdate)
      if nargin < 6
        tfForceUpdate = false;
      end
      %#%CALOK
      if (iFrm0 == iFrm1) && (iTgt0 == iTgt1),
        return;
      end

      if iFrm1 == obj.currFrameAdjust,
        % call is originating from checkAdjust failure
        return;
      end

      res = obj.checkAccept();
      if strcmpi(res,'Yes')
        % accept before going on
        obj.acceptLabels();
        obj.clearSelected();
      elseif strcmpi(res,'Cancel'),
        % go back a frame
        obj.labeler.setFrameAndTargetGUI(iFrm0,iTgt0);
        return;
      else % answer = No
        % pass
      end
      [tflabeled,lpos,lpostag] = obj.labeler.labelPosIsLabeled(iFrm1,iTgt1);
      if tflabeled
        obj.assignLabelCoords(lpos,'lblTags',lpostag);
        obj.enterAccepted(false);
      else
        assert(iTgt0==iTgt1,'Multiple targets unsupported.');
        assert(~obj.labeler.hasTrx,'Targets are unsupported.');
        obj.enterAdjust(true,false);
      end
      
      % working set: unchanged
      
%       obj.clearSelected();
      
      obj.projectionClear();
    end
    
    function clearLabels(obj)
      %#%CALOK
      obj.enterAdjust(true,true);
      obj.projectionWorkingSetClear();
      obj.projectionClear();
    end
    
    function acceptLabels(obj)
      obj.enterAccepted(true);
      obj.labeler.InitializePrevAxesTemplate();
    end
    
    function unAcceptLabels(obj)
      obj.enterAdjust(false,false);
    end 
    
    function axBDF(obj,src,evt) 
      
      if ~obj.labeler.isReady,
        return;
      end
      
      if evt.Button~=1
        % eg, Button==3 for pan
        return;
      end      
      
      iAx = find(src==obj.hAx);

      if obj.isPanZoom(iAx),
        return;
      end

      iWS = obj.iSetWorking;
      if ~isnan(iWS)
        iPt = obj.iSet2iPt(iWS,iAx);
        ax = obj.hAx(iAx);
        pos = get(ax,'CurrentPoint');
        pos = pos(1,1:2);
        obj.assignLabelCoordsIRaw(pos,iPt);
        obj.setPointAdjusted(iPt);
        
%         if ~obj.tfSel(iPt)
%           obj.clearSelected();
%           obj.toggleSelectPoint(iPt);
%         end
        
        if obj.tfOcc(iPt)
          obj.tfOcc(iPt) = false;
          obj.refreshOccludedPts();
        end
        % estOcc status unchanged
        
        if obj.streamlined && all(obj.tfAdjusted)
          obj.enterAccepted(true);
        else
          switch obj.state
            case LabelState.ADJUST
              % none
            case LabelState.ACCEPTED
              obj.enterAdjust(false,false);
          end
        end
        obj.projectionRefresh();
      end
    end
    
    function axOccBDF(obj,src,evt) %#ok<INUSD>
      
      if ~obj.labeler.isReady,
        return;
      end
      
      iAx = find(src==obj.hAxOcc);
      if obj.isPanZoom(iAx),
        return;
      end

      assert(isscalar(iAx));
      iWS = obj.iSetWorking;
      if ~isnan(iWS)
        iPt = obj.iSet2iPt(iWS,iAx); 
        obj.setPtFullOcc(iPt);
      end      
    end
    function setPtFullOcc(obj,iPt)
      obj.setPointAdjusted(iPt);
%       obj.clearSelected();
      
      obj.tfOcc(iPt) = true;
      obj.tfEstOcc(iPt) = false;
      obj.refreshOccludedPts();
      obj.refreshPtMarkers('iPts',iPt);
      
      if obj.streamlined && all(obj.tfAdjusted)
        obj.enterAccepted(true);
      else
        switch obj.state
          case LabelState.ADJUST
            % none
          case LabelState.ACCEPTED
            obj.enterAdjust(false,false);
        end
      end
      obj.projectionRefresh();
    end
    
    function ptBDF(obj,src,evt)
      
      if ~obj.labeler.isReady,
        return;
      end
      ax = get(src,'Parent');
      iAx = find(ax==obj.hAx);
      if obj.isPanZoom(iAx),
        return;
      end

      %#%CALOK
      iPt = src.UserData;
      switch evt.Button
        case 1
          obj.iPtMove = iPt;
          obj.tfMoved = false;
        case 3
          obj.toggleEstOccPoint(iPt);
      end
    end
    
    function wbmf(obj,src,evt) %#ok<INUSD>
      %#%CALOK
      
      if ~obj.labeler.isReady,
        return;
      end
      
      iPt = obj.iPtMove;
      if ~isnan(iPt)
        if obj.state==LabelState.ACCEPTED
          obj.enterAdjust(false,false);
        end

        iAx = obj.iPt2iAx(iPt);
        ax = obj.hAx(iAx);
        tmp = get(ax,'CurrentPoint');
        pos = tmp(1,1:2);
        obj.tfMoved = true;
        obj.assignLabelCoordsIRaw(pos,iPt);
        obj.setPointAdjusted(iPt);
        
        obj.projectionRefresh();
      end
    end
    
    function wbuf(obj,src,evt) %#ok<INUSD>
      %#%CALOK
      
      if ~obj.labeler.isReady,
        return;
      end
      
      iPt = obj.iPtMove;
      if ~isnan(iPt)
        obj.projectionRefresh();
      end
      obj.iPtMove = nan;
      obj.tfMoved = false;
    end
    
    function tfKPused = kpf(obj,src,evt) 
      
      if ~obj.labeler.isReady,
        return;
      end
      
      %#%CALOK
      key = evt.Key;
      modifier = evt.Modifier;      
      tfCtrl = any(strcmp('control',modifier));
      tfShft = any(strcmp('shift',modifier));
      
      lObj = obj.labeler;
      tfKPused = true;

      % shortcuts in main figure will be handled by menu items themselves
      if src == lObj.gdata.mainFigure_,
        match = [];
      else
        match = lObj.gdata.matchShortcut(evt);
      end

      if strcmp(key,'space')
        obj.toggleEpipolarState();
        %[tfSel,iSel] = obj.projectionPointSelected();
        %if tfSel && ~obj.tfOcc(iSel)
        %  obj.projectToggleState(iSel, -1);
        %end
      elseif strcmp(key,'s') && ~tfCtrl
        if obj.state==LabelState.ADJUST
          obj.acceptLabels();
        end
      elseif any(strcmp(key,{'d' 'equal'}))
        lObj.frameUpGUI(tfCtrl);
      elseif any(strcmp(key,{'a' 'hyphen'}))
        lObj.frameDownGUI(tfCtrl);
      elseif strcmp(key,'o') && ~tfCtrl
        [tfSel,iSel] = obj.projectionPointSelected();
        if tfSel
          obj.toggleEstOccPoint(iSel);
        end
      elseif strcmp(key,'u') && ~tfCtrl
        iAx = find(gcf==obj.hFig);
        iWS = obj.iSetWorking;        
        if isscalar(iAx) && ~isnan(iWS) && obj.labeler.showOccludedBox,
          iPt = obj.iSet2iPt(iWS,iAx);
          obj.setPtFullOcc(iPt);
        end
      elseif any(strcmp(key,{'leftarrow' 'rightarrow' 'uparrow' 'downarrow'}))
        %[tfSel,iSel] = obj.anyPointSelected();

        [tfSel,iSel,iAx] = obj.projectionPointSelected();
        if tfSel && ~obj.tfOcc(iSel)
          tfShift = any(strcmp('shift',modifier));
          xy = obj.getLabelCoordsI(iSel);
          %iAx = obj.iPt2iAx(iSel);
          ax = obj.hAx(iAx);
          switch key
            case 'leftarrow'
              xl = xlim(ax);
              dx = diff(xl);
              if tfShift
                xy(1) = xy(1) - dx/obj.DXFACBIG;
              else
                xy(1) = xy(1) - dx/obj.DXFAC;
              end
              %xy(1) = max(xy(1),1);
            case 'rightarrow'
              xl = xlim(ax);
              dx = diff(xl);
              if tfShift
                xy(1) = xy(1) + dx/obj.DXFACBIG;
              else
                xy(1) = xy(1) + dx/obj.DXFAC;
              end
              %ncs = lObj.movienc;
              %xy(1) = min(xy(1),ncs(iAx));
            case 'uparrow'
              yl = ylim(ax);
              dy = diff(yl);
              if tfShift
                xy(2) = xy(2) - dy/obj.DXFACBIG;
              else
                xy(2) = xy(2) - dy/obj.DXFAC;
              end
              %xy(2) = max(xy(2),1);
            case 'downarrow'
              yl = ylim(ax);
              dy = diff(yl);
              if tfShift
                xy(2) = xy(2) + dy/obj.DXFACBIG;
              else
                xy(2) = xy(2) + dy/obj.DXFAC;
              end
              %nrs = lObj.movienr;
              %xy(2) = min(xy(2),nrs(iAx));
          end
          obj.assignLabelCoordsIRaw(xy,iSel);
          switch obj.state
            case LabelState.ADJUST
              obj.setPointAdjusted(iSel);
            case LabelState.ACCEPTED
              obj.enterAdjust(false,false);
          end
          obj.projectionRefresh();
        else
          tfKPused = false;
        end
      elseif strcmp(key,'backquote')
        iSet = obj.numHotKeyPtSet+10;
        if iSet > obj.nPointSet
          iSet = 1;
        end
        obj.numHotKeyPtSet = iSet;
      elseif any(strcmp(key,{'0' '1' '2' '3' '4' '5' '6' '7' '8' '9'}))
        iSet = str2double(key);
        if iSet==0
          iSet = 10;
        end
        iSet = iSet+obj.numHotKeyPtSet-1;
        if iSet > obj.nPointSet
          % none
        else
          tfClearOnly = iSet==obj.iSetWorking;
          obj.projectionWorkingSetClear();
          obj.projectionClear();
%           obj.clearSelected();
          if ~tfClearOnly
            obj.projectionWorkingSetSet(iSet);
          end
        end
      elseif ~isempty(match),
        for i = 1:size(match,1),
          tag = match{i,1};
          cb = lObj.gdata.(tag).Callback;
          if ischar(cb),
            eval(cb);
          else
            cb(lObj.gdata.(tag),evt);
          end
        end

      else
        tfKPused = false;
      end
    end
    
    function shortcuts = LabelShortcuts(obj)

      shortcuts = cell(0,3);

      shortcuts{end+1,1} = 'Toggle whether epipolar lines are shown';
      shortcuts{end,2} = 'space';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = 'Accept current labels';
      shortcuts{end,2} = 's';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = 'Toggle whether selected kpt is occluded';
      shortcuts{end,2} = 'o';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = 'If fully-occluded box is shown, move selected keypoint to occluded box.';
      shortcuts{end,2} = 'u';
      shortcuts{end,3} = {};

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

      shortcuts{end+1,1} = sprintf('If kpt selected, move right by 1/%.1fth of axis size, ow forward one frame',obj.DXFAC);
      shortcuts{end,2} = 'Right arrow';
      shortcuts{end,3} = {};
      
      shortcuts{end+1,1} = sprintf('If kpt selected, move left by 1/%.1fth of axis size, ow back one frame',obj.DXFAC);
      shortcuts{end,2} = 'Left arrow';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = sprintf('If kpt selected, move up by 1/%.1fth of axis size',obj.DXFAC);
      shortcuts{end,2} = 'Up arrow';
      shortcuts{end,3} = {};
      
      shortcuts{end+1,1} = sprintf('If kpt selected, move down by 1/%.1f of axis size',obj.DXFAC);
      shortcuts{end,2} = 'Down arrow';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = sprintf('If kpt selected, move right by 1/%.1fth of axis size',obj.DXFACBIG);
      shortcuts{end,2} = '+';
      shortcuts{end,3} = {'Shift'};

      shortcuts{end+1,1} = sprintf('If kpt selected, move left by 1/%.1fth of axis size',obj.DXFACBIG);
      shortcuts{end,2} = '-';
      shortcuts{end,3} = {'Shift'};

      shortcuts{end+1,1} = sprintf('If kpt selected, move left by 1/%.1fth of axis size ow go to next %s',obj.DXFACBIG,...
        obj.labeler.movieShiftArrowNavMode.prettyStr);
      shortcuts{end,2} = 'Left arrow';
      shortcuts{end,3} = {'Shift'};

      shortcuts{end+1,1} = sprintf('If kpt selected, move right by 1/%.1fth of axis size, ow go to previous %s',obj.DXFACBIG,...
        obj.labeler.movieShiftArrowNavMode.prettyStr);
      shortcuts{end,2} = 'Right arrow';
      shortcuts{end,3} = {'Shift'};

      shortcuts{end+1,1} = sprintf('If kpt selected, move up by 1/%.1fth of axis size',obj.DXFACBIG);
      shortcuts{end,2} = 'Up arrow';
      shortcuts{end,3} = {'Shift'};

      shortcuts{end+1,1} = sprintf('If kpt selected, move down by 1/%.1fth of axis size',obj.DXFACBIG);
      shortcuts{end,2} = 'Down arrow';
      shortcuts{end,3} = {'Shift'};

      shortcuts{end+1,1} = 'Zoom in/out';
      shortcuts{end,2} = 'Mouse scroll';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = 'Pan view';
      shortcuts{end,2} = 'Mouse right-click-drag';
      shortcuts{end,3} = {};

    end


    function h = getLabelingHelp(obj) 
      h = cell(0,1);
      h{end+1} = 'Adjust all keypoints for all views, then click Accept to store. ';
      h{end+1} = '';
      h{end+1} = ['There is a set of template/"white" points on the ',...
        'image at all times. To ',...
        'label a frame, adjust the points as necessary and accept. Adjusted ',...
        'points are shown in colors (rather than white). '];
      h{end+1} = '';
      h{end+1} = ['Select a keypoint by typing the number identifying it. ',...
        'If you have more than 10 keypoints, the ` (backquote) key lets you ',...
        'change which set of 10 keypoints the number keys correspond to.'];
      h{end+1} = ['Once a keypoint is selected, it can be adjusted in any of ',...
        'the views by clicking on the desired location in an image. ',...
        'Fine adjustments can be made using the arrow keys. '];
      h{end+1} = ['Type the keypoint number again to deselect it, or type another ',...
        'keypoint number to select a different keypoint. '];
      h{end+1} = ['If no keypoints are selected, you can adjust any keypoint by ',...
        'clicking down on it and dragging it to the desired location.'];
      h{end+1} = '';
      h{end+1} = ['To convert from 2D keypoint coordinates in each view to a 3D ',...
        'coordinate, the cameras must be calibrated and calibration information must be ',...
        'loaded into the project. '];
      h{end+1} = ['If calibration information is provided, "epipolar" lines can be shown ',...
        'to help with labeling. If we know the 2-D coordinate of a point in one view, the ',...
        'epipolar line is the line that this point may lie on in another view. '];
      h{end+1} = ['Epipolar lines will be shown for keypoints after they are adjusted in any ',...
        'of the ways described above. An "a" will appear next to the keypoint number to ',...
        'indicate that its epipolar line is being shown. '];
      h{end+1} = 'You can toggle whether epipolar lines are shown or hidden with the space key.';
      h{end+1} = '';
      h1 = getLabelingHelp@LabelCore(obj);
      h = [h(:);h1(:)];

    end
    
    function refreshOccludedPts(obj)
      % Based on .tfOcc: 'Hide' occluded points in main image; arrange
      % occluded points in occluded box.
      
      if isempty(obj.hPtsOcc),
        return;
      end
      
      tf = obj.tfOcc;
      assert(isvector(tf) && numel(tf)==obj.nPts);
      nOcc = nnz(tf);
      iOcc = find(tf);
      obj.setPtsCoords(nan(nOcc,2),obj.hPts(tf),obj.hPtsTxt(tf));
      for iPt=iOcc(:)'
        iSet = obj.iPt2iSet(iPt);
        LabelCore.setPtsCoordsOcc([iSet 1],obj.hPtsOcc(iPt),obj.hPtsTxtOcc(iPt));
      end
      LabelCore.setPtsCoordsOcc(nan(obj.nPts-nOcc,2),...
        obj.hPtsOcc(~tf),obj.hPtsTxtOcc(~tf));
    end
        
  end
    
  methods % template
        
    function setRandomTemplate(obj)
      %#%CALOK
      
      lbler = obj.labeler;
      movctrs = lbler.movieroictr;
      movnrs = lbler.movienr;
      movncs = lbler.movienc;
      xy = nan(obj.nPts,2);
      for iPt=1:obj.nPts
        iAx = obj.iPt2iAx(iPt);
        nr = movnrs(iAx);
        nc = movncs(iAx);
        xy(iPt,1) = movctrs(iAx,1) + nc/3*2*(rand-0.5);
        xy(iPt,2) = movctrs(iAx,2) + nr/3*2*(rand-0.5);
      end
      obj.assignLabelCoords(xy,'tfClip',false);        
    end
    
  end
  
  methods
    
    function projectionWorkingSetClear(obj)
      h = obj.hPtsTxt;
      % KB 20181022 not sure why there is hPtsColors and
      % ptsPlotInfo.Colors, using obj.ptsPlotInfo.Colors
      hClrs = obj.ptsPlotInfo.Colors;
      %hClrs = obj.hPtsColors;
      for i=1:obj.nPts
        set(h(i),'Color',hClrs(i,:),'FontWeight','normal','EdgeColor','none');
      end
      set(obj.hPts,'HitTest','on');     
      obj.iSetWorking = nan;
      obj.labeler.currImHud.updateLblPoint(nan,obj.nPointSet);
    end
    
    function projectionWorkingSetSet(obj,iSet)
            
      iPtsSet = obj.iSet2iPt(iSet,:);

      h = obj.hPts;
      hPT = obj.hPtsTxt;
      % KB 20181022 not sure why there is hPtsColors and
      % ptsPlotInfo.Colors, using obj.ptsPlotInfo.Colors
      hClrs = obj.ptsPlotInfo.Colors;
      %hClrs = obj.hPtsColors;
      for i=1:obj.nPts
        if any(i==iPtsSet)
          set(hPT(i),'Color',hClrs(i,:),'FontWeight','bold','EdgeColor','w');
          set(h(i),'HitTest','on');
        else
          set(hPT(i),'Color',hClrs(i,:)*.75,'FontWeight','normal','EdgeColor','none');
          set(h(i),'HitTest','off');
        end
      end
      iAx = find(get(0,'CurrentFigure')==obj.hFig);
      if isscalar(iAx),
        set(obj.hFig(iAx),'CurrentObject',get(hPT(iAx),'Parent'));
      end
      obj.iSetWorking = iSet;
      obj.labeler.currImHud.updateLblPoint(iSet,obj.nPointSet);
      obj.projectionRefresh();
    end
    
    function projectionWorkingSetToggle(obj,iSet)
      if isnan(obj.iSetWorking)
        obj.projectionWorkingSetSet(iSet);
      else
        tfMatch = obj.iSetWorking==iSet;
        obj.projectionWorkingSetClear();
        if ~tfMatch
          obj.projectionWorkingSetSet(iSet);
        end
      end
    end
    
    function tf = projectionWorkingSetPointInWS(obj,iPt)
      % Returns true if iPt is in current working set.
      tf = obj.iPt2iSet(iPt)==obj.iSetWorking;
    end
    
    function [tfSel,iSelPt,iAx] = projectionPointSelected(obj)
      % AL 20211004. Instead of using the notion of "selected"-ness from 
      % LabelCore (.tfSel, .anyPointSelected, etc), use .iSetWorking and
      % the figure/view-with-focus. ie, if one of the APT figures/views has
      % focus, and a working set has been selected, then this implicitly
      % defines a point that is being worked on (or "selected").

      iAx = find(get(0,'CurrentFigure')==obj.hFig);
      iWS = obj.iSetWorking;
      tfSel = isscalar(iAx) && ~isnan(iWS);
      if tfSel
        iSelPt = obj.iSet2iPt(iWS,iAx);
      else
        iSelPt = nan;
      end
    end
    
    function projectionInit(obj)
      %obj.pjtIPts = nan(1, obj.nView);
      %obj.pjtIPts = [nan nan];
      %hLEpi = gobjects(1,obj.nView * (obj.nView - 1));
      hLEpi = gobjects(obj.nView, obj.nView);
      hLRcn = gobjects(1,obj.nView);
      ppimvcm = obj.ptsPlotInfo.MultiViewCalibratedMode;
      gdata = obj.labeler.gdata;
      for iV = 1:obj.nView
        ax = gdata.axes_all(iV);        
        
        % AL20160923 
        % EPlines (.pjtHLinesEpi) are sporadically not displaying on SH's
        % windows machine. Setting 'Marker' to '.' with 'MarkerSize' of 1
        % displays the line; using no Marker with 'LineStyle' of anything
        % doesn't show the line. No good explanation yet, when using
        % 'LineStyle' the display would periodically fail and then
        % re-appear on project restart. Best guess is low-level graphics
        % weirdness/bug. opengl('software') did not help.
        %
        % For now, include Marker with the line and see if it resolves.
        for j = 1:obj.nView
%           if iV == j
%             % skip when iV==j (the view for the clicked point is the same,
%             % there is no epipolar line.
%             continue
%           end
          hLEpi(iV, j) = plot(ax,nan,nan,'-',...
            'LineWidth',ppimvcm.EpipolarLineWidth,...
            'Marker','.',...
            'MarkerSize',1,...
            'PickableParts','none',...
            'Tag',sprintf('LabelCoreMV_Epi%d',iV));
        end
        hLRcn(iV) = plot(ax,nan,nan,ppimvcm.ReconstructedMarker,...
          'MarkerSize',ppimvcm.ReconstructedMarkerSize,...
          'LineWidth',ppimvcm.ReconstructedLineWidth,...
          'PickableParts','none',...
          'Tag',sprintf('LabelCoreMV_Rcn%d',iV));
      end
      obj.pjtHLinesEpi = hLEpi;
      obj.pjtHLinesRecon = hLRcn;
    end
    
    function projectionClear(obj)
      % Clear projection state.
      
      for iPt=1:obj.nPts
        set(obj.hPtsTxt(iPt),'String',obj.hPtsTxtStrs{iPt});
      end
      
      %obj.pjtIPts = nan(1, obj.nView);
      %obj.pjtIPts = [nan nan];
      % this will look odd, but make sure the visibilty is true. Although
      % the lines won't be visible due to the clearing. The next click will
      % make the lines visible again.
      obj.showEpiLines = true;

      % it is not enough to set all the lines to be invisible. Otherwise,
      % past clicked epipolor lines can become visible. So set all the
      % lines to NaN
      for i = 1:obj.nView
        for j = 1:obj.nView
          set(obj.pjtHLinesEpi(i, j),'XData',nan,'YData',nan,'Visible','off');
        end
      end
      
      %set(obj.pjtHLinesEpi,'Visible','off');
      set(obj.pjtHLinesRecon,'Visible','off');
      %obj.projectionWorkingSetClear();
    end
    
    function toggleEpipolarState(obj)
      if obj.showEpiLines == true
        obj.showEpiLines = false;
        set(obj.pjtHLinesEpi,'Visible','off');
      else
        obj.showEpiLines = true;
        set(obj.pjtHLinesEpi,'Visible','on');
      end
      
    end

    function projectionRefreshEPlines(obj)
      % update EPlines based on .pjtIPt1 and coords of that hPt.
      
      %assert(obj.pjtState==1);
      if ~obj.isCalRig,
        return;
      end

      % loop over the non nan points stored in pjtIPts, plot the epipolar line
      % for each of these points.
      for i = 1:length(obj.pjtIPts)
        if isnan(obj.pjtIPts(i))
          % it's probably safe to break at this point, because pjtIPts should be
          % filled from lowest to highest idx, but to be safe, we'll check the
          % rest of the pjtIPts'. This should be cleaned up as other refactoring
          % is done.
          % AL: seems like you need to keep looping bc .pjtIPts is indexed
          % by view, so what if eg view N had an anchor but the others
          % didnt?
          continue;
        end
        iPt1 = obj.pjtIPts(i);
        hPt1 = obj.hPts(iPt1);
        xy1 = [hPt1.XData hPt1.YData];
        iAx1 = obj.iPt2iAx(iPt1);
        %iAxOther = setdiff(1:obj.nView,iAx1);
        crig = obj.pjtCalRig;
        %for iAx = iAxOther
        % update text marker to include an a
        set(obj.hPtsTxt(iPt1),'String',[obj.hPtsTxtStrs{iPt1} 'a']);
        for iAx = 1:obj.nView
          if iAx == i
            continue
          end
          hIm = obj.hIms(iAx);
          imroi = [hIm.XData hIm.YData];
        
          [x,y] = crig.computeEpiPolarLine(iAx1,xy1,iAx,imroi);
        
          hEpi = obj.pjtHLinesEpi(iAx, i);
          set(hEpi,'XData',x,'YData',y,'Visible',onIff(obj.showCalibration),'Color',hPt1.Color);
          %fprintf('Epipolar line for axes %d should be visible, x = %s, y = %s\n',iAx,mat2str(x),mat2str(y));
        end
        %set(obj.pjtHLinesEpi(iAx1),'Visible','off');
      end
    end
    
    function projectionSet2nd(obj,iPt2)
      %#%CALOK
    
      assert(~isnan(obj.pjtIPts(1)));
      assert(isnan(obj.pjtIPts(2)));
      assert(iPt2~=obj.pjtIPts(1),'Second projection point must differ from anchor point.');
      %obj.pjtIPts(2) = iPt2;
      set(obj.pjtHLinesEpi,'Visible','off');
      
      set(obj.hPtsTxt(iPt2),'String',[obj.hPtsTxtStrs{iPt2} 'a']);
      
      obj.projectionRefreshReconPts();
    end
    
    function projectionRefreshReconPts(obj)
      % update recon pts based on .pjtIPt1 and .pjtIPt2 and coords of 
      % corresponding hPts.
      
      assert(obj.pjtState==2);
      if ~obj.isCalRig,
        return;
      end
      
      iPt1 = obj.pjtIPts(1);
      iPt2 = obj.pjtIPts(2);
      iAx1 = obj.iPt2iAx(iPt1);
      iAx2 = obj.iPt2iAx(iPt2);
      hPt1 = obj.hPts(iPt1);
      hPt2 = obj.hPts(iPt2);
      
      xy1 = [hPt1.XData hPt1.YData];
      xy2 = [hPt2.XData hPt2.YData];
      iAxOther = setdiff(1:obj.nView, [iAx1 iAx2]);
      crig = obj.pjtCalRig;
      for iAx = iAxOther
        [x,y] = crig.reconstruct(iAx1,xy1,iAx2,xy2,iAx);
        set(obj.pjtHLinesRecon(iAx),...
          'XData',x,'YData',y,...
          'Visible',onIff(obj.showCalibration),'Color',hPt1.Color);
      end
    end
    
    function projectionDisp3DPosn(obj)
      % Calculate/display triangulated 3D coordinates of current working
      % points (assuming all have been set/adjusted)
      
      iWS = obj.iSetWorking;
      if isnan(iWS)
        error('No current working pointset.');
      end
      
      ipts = obj.iSet2iPt(iWS,:);
      tfadj = obj.tfAdjusted(ipts);
      if ~all(tfadj)
        error('One or more points in current working pointset have not been set/adjusted.');
      end
      
      axs = obj.hAx;
      nvw = obj.nView;
      hPtsWS = obj.hPts(ipts);
      xs = cell2mat(get(hPtsWS,'XData'));
      ys = cell2mat(get(hPtsWS,'YData'));
      xyim = [xs(:) ys(:)]';
      xyim = reshape(xyim,[2 1 nvw]);
      
      crig = obj.pjtCalRig;
      [X,xyrp,rpe] = crig.triangulate(xyim);
      
      fprintf('Pointset %d (labelpoints %s)\n',iWS,mat2str(ipts));
      for ivw=1:nvw
        fprintf(' view %d: clicked posn is %s. RP err is %.2f\n',ivw,...
          mat2str(round(reshape(xyim(:,1,ivw),[1 2]))),rpe(ivw) );
      end
      fprintf('3D coordinates are %s\n',mat2str(X,5));
      
      % C+P LabelerGUI (commented out there)
      u0 = X(1);
      v0 = X(2);
      w0 = X(3);
      VIEWDISTFRAC = 5;
      obj.projectionClear3DPosn();
      for iview=1:nvw
        ax = axs(iview);
        
        % Start from where we want the 3D axes to be located in the view
        xl = ax.XLim;
        yl = ax.YLim;
        % Loop and find the scale where the the maximum projected length is ~
        % 1/8th the current view
        SCALEMIN = 0;
        SCALEMAX = 20;
        SCALEN = 300;
        avViewSz = (diff(xl)+diff(yl))/2;
        tgtDX = avViewSz/VIEWDISTFRAC*.8;
        scales = linspace(SCALEMIN,SCALEMAX,SCALEN);
        
        [x0,y0] = crig.project3d(u0,v0,w0,iview);
        for iScale = 1:SCALEN
          % origin is (u0,v0,w0) in 3D; (x0,y0) in 2D
          
          s = scales(iScale);
          [x1,y1] = crig.project3d(u0+s,v0,w0,iview);
          [x2,y2] = crig.project3d(u0,v0+s,w0,iview);
          [x3,y3] = crig.project3d(u0,v0,w0+s,iview);
          d1 = sqrt( (x1-x0).^2 + (y1-y0).^2 );
          d2 = sqrt( (x2-x0).^2 + (y2-y0).^2 );
          d3 = sqrt( (x3-x0).^2 + (y3-y0).^2 );
          if d1>tgtDX || d2>tgtDX || d3>tgtDX
            fprintf(1,'Found scale: %.2f\n',s);
            break;
          end
        end
        
        LINEWIDTH = 5;
        FONTSIZE = 22;
        lineargs = {'LineWidth' LINEWIDTH};
        textargs = {'fontweight' 'bold' 'fontsize' FONTSIZE 'parent' ax};
        obj.pjtShow3D(end+1,1) = plot(ax,[x0 x1],[y0 y1],'r-',lineargs{:});
        obj.pjtShow3D(end+1,1) = text(x1,y1,'x','Color',[1 0 0],textargs{:});
        obj.pjtShow3D(end+1,1) = plot(ax,[x0 x2],[y0 y2],'g-',lineargs{:});
        obj.pjtShow3D(end+1,1) = text(x2,y2,'y','Color',[0 0.5 0],textargs{:});
        obj.pjtShow3D(end+1,1) = plot(ax,[x0 x3],[y0 y3],'-',...
          'Color',[0 1 1],lineargs{:});
        obj.pjtShow3D(end+1,1) = text(x3,y3,'z','Color',[0 0 1],textargs{:});
      end
    end
    function projectionClear3DPosn(obj)
      deleteValidGraphicsHandles(obj.pjtShow3D);
      obj.pjtShow3D = gobjects(0,1);
    end
    
    function projectionRefresh(obj)
      obj.projectionRefreshEPlines();
%       switch obj.pjtState
%         case 0
%           % none
%         case 1
%           obj.projectionRefreshEPlines();
%         case 2
%           obj.projectionRefreshReconPts();
%         otherwise
%           assert(false);
%       end
    end
    
    function projectionSetCalRig(obj,crig)
      assert(isa(crig,'CalRig'));
      
      %20160923 hack legacy CalRigSH objs and EPlines workaround
      if isa(crig,'CalRigSH')
        %crig.epLineNPts = 1e4;
        crig.epLineNPts = 2;
      end 
      
%       crigRois = crig.viewRois; % [nView x 4]; each row is [xlo xhi ylo yhi]
%       lObj = obj.labeler;
%       imsAll = lObj.gdata.images_all;
%       for iView = 1:obj.nView
%         xdata = imsAll(iView).XData;
%         ydata = imsAll(iView).YData;
%         imroi = [xdata ydata];
%         crigroi = crigRois(iView,:);
%         if ~isequal(imroi,crigroi)
%           warningNoTrace('LabelCoreMultiViewCalibrated:projectionSetCalRig',...
%             'View %d (%s): image roi is %s; calibration roi is %s',...
%             iView,crig.viewNames{iView},mat2str(imroi,6),mat2str(crigroi,6));
%         end
%       end
      
      obj.pjtCalRig = crig;
    end
    
  end
  
  methods
    
    % ADJUST/ACCEPTED-NESS
    % What is the "Adjust" state?
    % - The button says "Accept" => Either
    % - 1) The current frame has no recorded labels (all pts shown as
    %       white) OR
    % - 2) The current frame HAS recorded labels (pts shown in color), but 
    %       they differ from the points as currently shown 
    %
    % Meanwhile, in the "Accept" state:
    % - The Button says "Accepted" =>
    % - 1) What you see is what is recorded in the Labeler (pts shown in
    %   color)
	%
    % Regardless of projection state, you can Accept, which writes
    % current positions to Labeler.

    function v = isUnsavedState(obj)
      v = any(obj.tfAdjusted & ~obj.tfStored);
    end

    function v = isAdjustFrameChange(obj)
      v = ~isnan(obj.currFrameAdjust) && (obj.currFrameAdjust ~= obj.labeler.currFrame);
    end
    
    function res = checkAccept(obj)
      res = 'No';
      if obj.isUnsavedState() && obj.isAdjustFrameChange(),
        res = questdlg('Some keypoints have been adjusted but not accepted. Accept before losing this information?','Accept labels','Yes','No','Cancel','Yes');
        figure(gcf); % somehow focus gets lost after question, and keyboard shortcuts don't work...
      end

    end

    function enterAdjust(obj,tfResetPts,tfClearLabeledPos)
      % Enter adjustment state for current frame/tgt.
      %
      % if tfReset, reset all points to pre-adjustment (white).
      % if tfClearLabeledPos, clear labeled pos.

      %#%CALOKedit LabelCoreMul
      if tfResetPts
        obj.tfEstOcc(:) = 0; % reset all points to NOT be occluded
        if obj.streamlined
          [obj.hPts.XData] = deal(nan);
          [obj.hPts.YData] = deal(nan);
          [obj.hPtsTxt.Position] = deal([nan nan 1]);
        else
          tpClr = obj.ptsPlotInfo.TemplateMode.TemplatePointColor;
          obj.refreshPtMarkers();
          arrayfun(@(x)set(x,'Color',tpClr),obj.hPts);
        end
        obj.tfAdjusted(:) = false;
        obj.tfStored(:) = false;
        obj.currFrameAdjust = obj.labeler.currFrame;
      end
      if tfClearLabeledPos
        obj.labeler.labelPosClear();
      end
        
      obj.iPtMove = nan;
      obj.tfMoved = false;
      
      set(obj.tbAccept,'BackgroundColor',[0.6,0,0],'String','Accept',...
        'Value',0,'Enable','on');
      obj.currFrameAdjust = obj.labeler.currFrame;
      obj.state = LabelState.ADJUST;
    end
        
    function enterAccepted(obj,tfSetLabelPos)
      % Enter accepted state for current frame/tgt. All points colored. If
      % tfSetLabelPos, all points/tags written to labelpos/labelpostag.
      
      %#%CALOK
            
      nPts = obj.nPts;
      ptsH = obj.hPts;
      % KB 20181022 not sure why there is hPtsColors and
      % ptsPlotInfo.Colors, using obj.ptsPlotInfo.Colors
      clrs = obj.ptsPlotInfo.Colors;
      %clrs = obj.hPtsColors;
      for i = 1:nPts
        set(ptsH(i),'Color',clrs(i,:));
        if ~isempty(obj.hPtsOcc),
          set(obj.hPtsOcc(i),'Color',clrs(i,:));
        end
      end
      
      obj.tfAdjusted(:) = true;
      obj.tfStored(:) = true;
      obj.currFrameAdjust = nan;
      
      if tfSetLabelPos
        xy = obj.getLabelCoords();
        obj.labeler.labelPosSet(xy);
        obj.setLabelPosTagFromEstOcc();
      end
      set(obj.tbAccept,'BackgroundColor',[0,0.4,0],'String','Accepted',...
        'Value',1,'Enable','on');

      obj.currFrameAdjust = nan;
      obj.state = LabelState.ACCEPTED;
    end
    
    function setPointAdjusted(obj,iSel)     
      obj.tfStored(iSel) = false;
      if ~obj.tfAdjusted(iSel)
        obj.tfAdjusted(iSel) = true;
        % KB 20181022 not sure why there is hPtsColors and
        % ptsPlotInfo.Colors, using obj.ptsPlotInfo.Colors
        clr = obj.ptsPlotInfo.Colors(iSel,:);
        %clr = obj.hPtsColors(iSel,:);
        set(obj.hPts(iSel),'Color',clr);
        if ~isempty(obj.hPtsOcc),
          set(obj.hPtsOcc(iSel),'Color',clr);
        end
      end
    end
    
    function toggleEstOccPoint(obj,iPt)
      obj.tfEstOcc(iPt) = ~obj.tfEstOcc(iPt);
      obj.refreshPtMarkers('iPts',iPt);
      if obj.state==LabelState.ACCEPTED
        obj.enterAdjust(false,false);
      end
    end
        
    function refreshHotkeyDesc(obj)
      iSet0 = obj.numHotKeyPtSet;
      iSet1 = iSet0+9;
      str = sprintf('Hotkeys 1-9,0 map to 3d points %d-%d, ` (backquote) toggles',iSet0,iSet1);
      [obj.hAxXLabels(2:end).String] = deal(str);
      obj.txLblCoreAux.String = str;
    end
    
    function toggleShowCalibration(obj)
      
      obj.setShowCalibration(~obj.showCalibration);
      
    end
            
  end
  
end
