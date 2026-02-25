classdef LabelCoreTemplate < LabelCore
% Template-based labeling  
  
  % DESCRIPTION
  % In Template mode, there is a set of template/"white" points on the
  % image at all times. (When starting, these points are randomized.) To 
  % label a frame, adjust the points as necessary and accept. Adjusted 
  % points are shown in colors (rather than white).
  %
  % Points may also be Selected using hotkeys (0..9). When a point is
  % selected, the arrow-keys adjust the point as if by mouse. Mouse-clicks
  % on the image also jump the point immediately to that location.
  % Adjustment of a point in this way is identical in concept to
  % click-dragging.
  %
  % To toggle a point as EstimatedOccluded, right-click on the point, or
  % use the 'o' hotkey.
  
  % DETAILED DESC
  % Indiv points, Adjustedness. 
  %  Unadjusted points. These have not been touched by a human. If tracking 
  %   results do not exist, these points are shown in white and are placed 
  %   at guesstimates to their true location. 
  % 
  %   If tracking results do exist, then these points are currently shown
  %   with a larger marker and thinner linestyle.
  %
  %   For now, unadjusted points can be selected or estocc'ed and this does
  %   not alter their unadjusted-ness. ESTOCC SHOULD ADJUST
  %  Adjusted points. These have been click-dragged,
  %   selected-and-mouseclicked, or selected-and-arrow-moved. Their
  %   unadjusted-modification to point cosmetics is removed. Adjusted
  %   points may still be selected.
  %
  % Overall state, Adjust vs Accepted.
  %  The current frame can be in an overall Accepted or overall Adjusting
  %  state. In the Accepted state, the "Accept" button is depressed and all
  %  points appear as if adjusted. Points can still be selected/est-occ'ed 
  %  without exiting this state. OCCING SHOULD ENTER ADJUST
  %
  %  The alternative overall state is Adjusting, where the labels/points
  %  differ in some respect from what is "recorded in the "DB". In this
  %  state, navigation to another frame and returning will result in
  %  reversion of the points to their stored state. To record the
  %  adjustments, the user must press Accept.
  
  % MARKER COSMETICS 
  %
  % The baseline cosmetics are stored in .ptsPlotInfo (ppi) which is 
  % initted from lObj.labelPointsPlotInfo. Currently LabelCoreTemplate does 
  % not consider lObj.predPointsPlotInfo even though it does display 
  % tracking; the tracking is shown as a labeling aide.
  %
  % Marker. The .Marker prop is affected by:
  %   * Selected-ness: 0 or 1. 
  %     - 1 currently hardcoded to use 'x'
  %   * Est-occluded-ness: 0 or 1
  %     - 1 currently hardcoded to 'o'; 's' for Selected&EstOcc
  %
  % MarkerColor. The marker Color is currently only affected by
  % Adjusted-ness; unadj-raw are white and everything else is per ppi.
  % 
  % OtherMarkerProps (Size, LineWidth). This is affected by:
  %   * Adjusted-ness: {adj, unadj-raw, unadj-w-trk}
  %     - unadj-raw currently is the same as adj.
  %     - unadj-w-trk uses a larger/thinner marker.
  %
  % Marker Text Label: Currently only the .FontAngle mutates; it is
  %   italicized for unadj-w-trk.

  % SUB APIS
  %
  % Adjustedness is stored in .tfAdjusted and should be mutated only with
  % the adjusted* api. 
  % (Adjustedness affects {Color, OtherMarkerProps, MarkerTextLabel})
  %
  % Selectedness.
  % A selected point is one that is currently being "worked on". This gives
  % the point "focus" and enables eg using arrow keys for adjustment. Only 
  % one point can be selected at a time. 
  %
  % Selectedness is stored in .tfSel and should be mutated only through the
  % LabelCore/selected* api. Selectedness is indicated in the UI via hPt
  % markers; use refreshPtMarkers() to update the UI.
  %
  % Estimated-occluded-ness.
  % An estimated-occluded point is one whose position is uncertain to some
  % degree due to occlusions in the image. 
  %
  % Estimated-occ-ness is stored in .tfEstOcc and should be mutated only
  % through toggleEstOcc. Est-occ-ness is indicated in the UI via hPt
  % markers, and again refreshPtMarkers() is the universal update.
  %
  % This business is pretty complicated and the factoring between LCT and
  % LC seems somehow suboptimal or outdated at this pt.
  % Marker <= (affected by) Sel, EO
  % MarkerProps: <= tfAdj, hasTrk
  % Color <= tfAdj
  % MarkerText <= tfAdj, hasTrk
  % Coords => LabelCore
  % Selected => LabelCore API => Marker
  % Adj => LabelCoreTemplate API => MarkerProps, Color, MarkerTxt
  % EO => LabelCore => Marker
  
  % --- OLDER NOTES BELOW ---
  % IMPL NOTES
  % 2 basic states, Adjusting and Accepted
  % - Adjusting, unlabeled frame underneath
  % -- 'untouched' template points are white
  % -- 'touched' template points are colored
  % -- Accept writes to labeledpos
  % - Adjusting, labeled frame underneath
  % -- all pts colored/'touched'
  % -- labeledpos exists underneath and is not overwritten until Acceptance
  % - Accepted
  % -- all points colored. 
  % -- Can still adjust, and adjustments will directly be stored to
  % labeledpos
  %
  % TRANSITIONS W/OUT TRX
  % Note: Once a template is created/loaded, there is a set of points (either
  %  all white/template, or partially template, or all colored) on the image
  %  at all times.
  % - New unlabeled frame
  % -- Label point positions are unchanged, but all points become white.
  %    If scrolling forward, the template from frame-1 will be used etc.
  % -- (Enhancement? not implemented) Accepted points from 'nearest'
  %    neighboring labeled frame (eg frm-1, or <frmLastVisited>, become
  %    white points in new frame
  % - New labeled frame
  % -- Previously accepted labels shown as colored points.
  %
  % TRANSITIONS W/TRX
  % - New unlabeled frame (current target)
  % -- Existing points are auto-aligned onto new frame. The existing
  %    points could represent previously accepted labels, or just the
  %    previous template state.
  % - New labeled frame (current target)
  % -- Previously accepted labels shown as colored points.
  % - New target (same frame), unlabeled
  % -- Last labeled frame for new target acts as template; points auto-aligned.
  % -- If no previously labeled frame for this target, current white points are aligned onto new target.
  % - New target (same frame), labeled
  % -- Previously accepted labels shown as colored points.
  
  

  properties
    supportsSingleView = true;
    supportsMultiView = false;
    supportsCalibration = false;
    supportsMultiAnimal = false;
  end
  
  properties
    iPtMove;     % scalar. Either nan, or index of pt being moved
    tfMoved;     % scalar logical; if true, pt being moved was actually moved
    
    tfAdjusted;  % nPts x 1 logical vec. If true, pt has been adjusted from 
                 % template or tracking prediction
                 
    % scalar LabelCoreTemplateResetType
    % The only way to clear/reset point(s) to unadjusted is to call     
    % setAllPointsUnadjusted. At this time, a LabelCoreTemplateResetType is
    % passed essentially indicating whether the frame has existing
    % tracking/predictions or not. We record this, the point being that if
    % at any time an element of .tfAdjusted is false, this property
    % indicates whether tracking is present for that unadjusted point.
    lastSetAllUnadjustedResetType = LabelCoreTemplateResetType.RESET
    
    %kpfIPtFor1Key;  % scalar positive integer. This is the point index that 
                 % the '1' hotkey maps to, eg typically this will take the 
                 % values 1, 11, 21, ...
                 
    % See cosmetics discussion above. Predicted-unadjustedness (or not)
    % toggles Marker Color, Other Marker Props, and Marker Txt Angle
    hPtsMarkerPVPredUnadjusted; % HG PV-pairs for unadjusted tracking predictions
    hPtsMarkerPVNotPredUnadjusted; % HG PV-pairs for not that; reverts the above
    hPtsTxtPVPredUnadjusted % etc
    hPtsTxtPVNotPredUnadjusted
  end
  
  properties
    unsupportedKPFFns = {} ;  % cell array of field names for objects that have general keypressfcn 
                              % callbacks but are not supported for this LabelCore
  end
  
  methods
    
    function obj = LabelCoreTemplate(varargin)
      obj = obj@LabelCore(varargin{:});
    end
    
    function initHook(obj)
      npts = obj.nPts;
      obj.tfAdjusted = false(npts,1);
      
      obj.updatePredUnadjustedPVs();
      
      obj.txLblCoreAux.Visible = 'on';
      obj.kpfIPtFor1Key = 1;
      obj.refreshTxLabelCoreAux();
      
      obj.state = LabelState.ADJUST; % AL 20190123 semi-hack. init to something/anything to avoid error at projectload/inithell

      % LabelCore should prob not talk directly to tracker
      tObj = obj.labeler.tracker;
      if ~isempty(tObj) && ~tObj.hideViz
        warningNoTrace('LabelCoreTemplate:viz',...
          'Enabling View>Hide Predictions. Tracking predictions (when present) are now shown as template points in Template Mode.');
        tObj.setHideViz(true);
      end
    end
    
  end
  
  methods

    % For LabelCoreTemplate, newFrameAndTarget() combines all the brains of
    % transitions for convenience reasons
    
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
      lObj = obj.labeler;
      
      [tflabeled,lpos,lpostag] = lObj.labelPosIsLabeled(iFrm1,iTgt1);
      if tflabeled
        obj.assignLabelCoords(lpos,'lblTags',lpostag);
        obj.enterAccepted(false);
        return;
      end
      
      assert(iFrm1==lObj.currFrame);
      [tftrked,lposTrk,occTrk] = lObj.trackIsCurrMovFrmTracked(iTgt1);
      if tftrked
        obj.assignLabelCoords(lposTrk,'lblTags',occTrk);
        obj.enterAdjust(LabelCoreTemplateResetType.RESETPREDICTED,false);
        return;
      end
      
      [tflbl2,lpos2] = lObj.labels2IsCurrMovFrmLbled(iTgt1);
      if tflbl2
        occ2 = false(size(lpos2,1),1); % currently labeledpos2 has no occ/tag fld
        obj.assignLabelCoords(lpos2,'lblTags',occ2);
        obj.enterAdjust(LabelCoreTemplateResetType.RESETPREDICTED,false);
        return;
      end      
      
      if iTgt0==iTgt1 % same target, new frame
        if lObj.hasTrx
          % existing points are aligned onto new frame based on trx at
          % (currTarget,prevFrame) and (currTarget,currFrame)
          
          xy0 = obj.getLabelCoords();
          xy = LabelCore.transformPtsTrx(xy0,...
            lObj.trx(iTgt0),iFrm0,...
            lObj.trx(iTgt0),iFrm1);
          obj.assignLabelCoords(xy,'tfClip',true);
        else
          % none, leave pts as-is
        end
      else % different target
        assert(lObj.hasTrx,'Must have trx to change targets.');
        [tfneighbor,iFrm0Neighb,lpos0] = ...
          lObj.labelPosLabeledNeighbor(iFrm1,iTgt1);
        if tfneighbor
          lpos0 = reshape(lpos0,[],2);
          xy = LabelCore.transformPtsTrx(lpos0,...
            lObj.trx(iTgt1),iFrm0Neighb,...
            lObj.trx(iTgt1),iFrm1);
        else
          % no neighboring previously labeled points for new target.
          % Just start with current points for previous target/frame.
          xy0 = obj.getLabelCoords();
          xy = LabelCore.transformPtsTrx(xy0,...
            lObj.trx(iTgt0),iFrm0,...
            lObj.trx(iTgt1),iFrm1);
        end
        obj.assignLabelCoords(xy,'tfClip',true);
      end
      obj.enterAdjust(LabelCoreTemplateResetType.RESET,false);
    end
    
    function clearLabels(obj)
      obj.clearSelected();
      obj.enterAdjust(LabelCoreTemplateResetType.RESET,true);
    end
    
    function acceptLabels(obj)
      obj.enterAccepted(true);
      % notify(obj.labeler, 'initializePrevAxesTemplate');
      obj.labeler.restorePrevAxesMode() ;
    end
    
    function unAcceptLabels(obj)
      assert(false,'Unsupported');
      obj.enterAdjust(LabelCoreTemplateResetType.NORESET,false);
    end 
    
    function axBDF(obj,src,evt) 
      
      if ~obj.labeler.isReady || evt.Button>1
        return;
      end
      if obj.isPanZoom(),
        return;
      end

      [tf,iSel] = obj.anyPointSelected();
      if tf
        pos = get(obj.hAx,'CurrentPoint');
        pos = pos(1,1:2);
        obj.assignLabelCoordsIRaw(pos,iSel);
        obj.setPointAdjusted(iSel);
        obj.toggleSelectPoint(iSel);
        if obj.tfOcc(iSel)
          obj.tfOcc(iSel) = false;
          obj.refreshOccludedPts();
        end
        % estOcc status unchanged
        switch obj.state
          case LabelState.ADJUST
            % none
          case LabelState.ACCEPTED
            obj.storeLabels();
            % KB 20181029: adjustments push directly to labeledpos
            %obj.enterAdjust(LabelCoreTemplateResetType.NORESET,false);
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
      
      mod = obj.hFig.CurrentModifier;
      tfShift = any(strcmp(mod,'shift'));
      if ~tfShift
        iPt = get(src,'UserData');
        obj.toggleSelectPoint(iPt);
        % prepare for click-drag of pt
        
        if obj.state==LabelState.ACCEPTED
          % KB 20181029
          %obj.enterAdjust(LabelCoreTemplateResetType.NORESET,false);
        end
        obj.iPtMove = iPt;
        obj.tfMoved = false;
      else
        iPt = get(src,'UserData');
        obj.toggleEstOccPoint(iPt);
      end
    end
    
    function wbmf(obj,src,evt) %#ok<INUSD>
      
      if ~obj.labeler.isReady,
        return;
      end
      
      if obj.state==LabelState.ADJUST || obj.state==LabelState.ACCEPTED
        iPt = obj.iPtMove;
        if ~isnan(iPt)
          ax = obj.hAx;
          tmp = get(ax,'CurrentPoint');
          pos = tmp(1,1:2);
          obj.tfMoved = true;
          obj.assignLabelCoordsIRaw(pos,iPt);
          obj.setPointAdjusted(iPt);
        end
      end
    end
    
    function wbuf(obj,src,evt) %#ok<INUSD>
      
      if ~obj.labeler.isReady,
        return;
      end
      
      if obj.state==LabelState.ADJUST || obj.state==LabelState.ACCEPTED,
        iPt = obj.iPtMove;
        if ~isnan(iPt) && ~obj.tfMoved
          % point was clicked but not moved
          obj.clearSelected(iPt);
          obj.toggleSelectPoint(iPt);
        end
        
        obj.iPtMove = nan;
        if obj.state==LabelState.ACCEPTED && ~isnan(iPt) && obj.tfMoved,
          obj.storeLabels();
        end
        obj.tfMoved = false;
      end
    end
    
    function tfKPused = kpf(obj,src,evt)
      
      if ~obj.labeler.isReady,
        return;
      end
      
      key = evt.Key;
      modifier = evt.Modifier;
      tfCtrl = any(strcmp('control',modifier));
      tfShft = any(strcmp('shift',modifier));
      
      tfKPused = true;
      lObj = obj.labeler;
      
      % Hack iss#58. Ensure focus is not on slider_frame. In practice this
      % callback is called before slider_frame_Callback when slider_frame
      % has focus.
      txStatus = lObj.gdata.txStatus;
      if src~=txStatus % protect against repeated kpfs (eg scrolling vid)
        uicontrol(lObj.gdata.txStatus);
      end

      if any(strcmp(key,{'s' 'space'})) && ~tfCtrl
        if obj.state==LabelState.ADJUST
          obj.acceptLabels();
        end
      elseif any(strcmp(key,{'d' 'equal'}))
        obj.controller.frameUp(tfCtrl);
      elseif any(strcmp(key,{'a' 'hyphen'}))
        obj.controller.frameDown(tfCtrl);
      elseif strcmp(key,'o') && ~tfCtrl
        [tfSel,iSel] = obj.anyPointSelected();
        if tfSel
          obj.toggleEstOccPoint(iSel);
        end
      elseif any(strcmp(key,{'leftarrow' 'rightarrow' 'uparrow' 'downarrow'}))
        [tfSel,iSel] = obj.anyPointSelected();
        if tfSel && ~obj.tfOcc(iSel)
          tfShift = any(strcmp('shift',modifier));
          xy = obj.getLabelCoordsI(iSel);
          switch key
            case 'leftarrow'
              dxdy = -obj.controller.videoCurrentRightVec();
            case 'rightarrow'
              dxdy = obj.controller.videoCurrentRightVec();
            case 'uparrow'
              dxdy = obj.controller.videoCurrentUpVec();
            case 'downarrow'
              dxdy = -obj.controller.videoCurrentUpVec();
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
              obj.setPointAdjusted(iSel);
            case LabelState.ACCEPTED
              obj.enterAdjust(LabelCoreTemplateResetType.NORESET,false);
          end
        else
          tfKPused = false;
        end
      elseif strcmp(key,'backquote')
        iPt = obj.kpfIPtFor1Key+10;
        if iPt > obj.nPts
          iPt = 1;
        end
        obj.kpfIPtFor1Key = iPt;
      elseif any(strcmp(key,{'0' '1' '2' '3' '4' '5' '6' '7' '8' '9'}))
        iPt = str2double(key);
        if iPt==0
          iPt = 10;
        end
        iPt = iPt+obj.kpfIPtFor1Key-1;
        if iPt > obj.nPts
          return;
        end
        %obj.clearSelected(iPt);
        obj.toggleSelectPoint(iPt);
      else
        tfKPused = false;
      end
    end
    
    function axOccBDF(obj,src,evt) %#ok<INUSD>
      % Note: currently occluded axis hidden so this should be uncalled
      
      if ~obj.labeler.isReady,
        return;
      end
      if obj.isPanZoom(),
        return;
      end

      
      [tf,iSel] = obj.anyPointSelected();
      if tf
        obj.setPointAdjusted(iSel);
        obj.toggleSelectPoint(iSel);
        obj.tfOcc(iSel) = true;
        obj.tfEstOcc(iSel) = false;
        obj.refreshOccludedPts();
        obj.refreshPtMarkers('iPts',iSel);
        switch obj.state
          case LabelState.ADJUST
            % none
          case LabelState.ACCEPTED
            obj.enterAdjust(LabelCoreTemplateResetType.NORESET,false);
        end
      end   
    end

    function shortcuts = LabelShortcuts(obj)

      shortcuts = cell(0,3);

      shortcuts{end+1,1} = 'Accept current labels';
      shortcuts{end,2} = 's or space';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = 'Toggle whether selected kpt is occluded';
      shortcuts{end,2} = 'o';
      shortcuts{end,3} = {};

      shortcuts{end+1,1} = 'Toggle whether selected kpt is fully occluded';
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

    function h = getLabelingHelp(obj) 
      h = cell(0,1);
      h{end+1} = 'Adjust all keypoints, then click Accept to store.';
      h{end+1} = '';
      h{end+1} = ['In Template labeling mode, there is a set of template/"white" points on the ',...
      'image at all times. To ',...
      'label a frame, adjust the points as necessary and accept. Adjusted ',...
      'points are shown in colors (rather than white). '];
      h{end+1} = ['Points may also be Selected using hotkeys (0..9). When a point is ',...
        'selected, the arrow-keys adjust the point as if by mouse. Mouse-clicks ',...
        'on the image also jump the point immediately to that location. '];
      h{end+1} = 'If no point is selected, you can click and drag a point to move it. ';
      h{end+1} = '';
      h{end+1} = ['Once you have finished adjusting all points, click the Accept button ',...
        'to store the coordinates. If you change frames before accepting, your work will be lost.'];
      h{end+1} = ['You can adjust points once they are accepted. If you change the points, you ',...
        'must click the Accept button again to store your work.'];
      h{end+1} = '';
      h1 = getLabelingHelp@LabelCore(obj);
      h = [h(:);h1(:)];
    end
            
  end
  
  methods % Cosmetics related; LabelCore cosmetics overloads
    
    % These overloads exist so that Templatemode-specific cosmetic state
    % can be updated when core cosmetic state (.ptsPlotInfo) is updated.
    
    function updatePredUnadjustedPVs(obj)
      % update .hPts*PV*Unadjusted from .ptsPlotInfo
          
      % Currently we hardcode this but could change in future

      ppi = obj.ptsPlotInfo;
      obj.hPtsMarkerPVPredUnadjusted = struct( ...
        'MarkerSize',ppi.MarkerProps.MarkerSize*1.5,...
        'LineWidth',ppi.MarkerProps.LineWidth/2);
      obj.hPtsMarkerPVNotPredUnadjusted = struct( ...
        'MarkerSize',ppi.MarkerProps.MarkerSize,...
        'LineWidth',ppi.MarkerProps.LineWidth);
      obj.hPtsTxtPVPredUnadjusted = struct( ...
        'FontAngle','italics');
      obj.hPtsTxtPVNotPredUnadjusted = struct( ...
        'FontAngle',ppi.TextProps.FontAngle);      
    
      % 20191018 AL maybe useful pattern later in future
%       predvizMrkProps = ppi.MarkerProps;
%       predvizMrkProps.MarkerSize = 1.5*predvizMrkProps.MarkerSize;
%       predvizMrkProps.LineWidth = predvizMrkProps.LineWidth/2;
%       predvizTxtProps = ppi.TextProps;
%       predvizTxtProps.FontAngle = 'italic';
%       
%       ppi.TemplateMode.PredViz.MarkerProps = predvizMrkProps;
%       ppi.TemplateMode.PredViz.TextProps = predvizTxtProps;
%       obj.ptsPlotInfo = ppi;
    end
    
    function updateColors(obj,colors)
      % LabelCore overload

      obj.ptsPlotInfo.Colors = colors;

      % Set colors for adjusted pts or unadj/predicted
      resetType = obj.lastSetAllUnadjustedResetType;
      tfSetColor = obj.tfAdjusted | ...
                   resetType==LabelCoreTemplateResetType.RESETPREDICTED;
      for i = 1:obj.nPts
        if tfSetColor(i)
          if numel(obj.hPts) >= i && ishandle(obj.hPts(i)),
            set(obj.hPts(i),'Color',colors(i,:));
          end
          if numel(obj.hPtsTxt) >= i && ishandle(obj.hPtsTxt(i)),
            set(obj.hPtsTxt(i),'Color',colors(i,:));
          end
          % Occ pts currently hidden
          %         if numel(obj.hPtsOcc) >= i && ishandle(obj.hPtsOcc(i)),
          %           set(obj.hPtsOcc(i),'Color',obj.ptsPlotInfo.Colors(i,:));
          %         end
          %         if numel(obj.hPtsTxtOcc) >= i && ishandle(obj.hPtsTxtOcc(i)),
          %           set(obj.hPtsTxtOcc(i),'Color',obj.ptsPlotInfo.Colors(i,:));
          %         end

          % TO DO uncomment
          % for i = 1:numel(obj.hSkel),
          % color = obj.ptsPlotInfo.Colors(obj.labeler.skeletonEdgeColor(i),:);
          % set(obj.hSkel(i),'Color',color);
          % end
        end
      end
    end
    
    function updateMarkerCosmetics(obj,pvMarker)
      flds = fieldnames(pvMarker);
      for f=flds(:)',f=f{1}; %#ok<FXSET>
        obj.ptsPlotInfo.MarkerProps.(f) = pvMarker.(f);
      end
      
      obj.updatePredUnadjustedPVs();
      
      obj.refreshPtMarkers(); % updates hPts.Marker
      obj.refreshMarkerProps(); % updates other marker-related props
    end
    
    function updateTextLabelCosmetics(obj,pvText,txtoffset)
      % Currently, if pvText includes FontAngle this will collide with 
      % Unadjusted/Predicted-ness and this will not be handled properly.
      % (However FontAngle currently *not* exposed in cosmetics picker)
      
      flds = fieldnames(pvText);
      for f=flds(:)',f=f{1}; %#ok<FXSET>
        obj.ptsPlotInfo.TextProps.(f) = pvText.(f);
      end
      set(obj.hPtsTxt,pvText);
      
      obj.ptsPlotInfo.TextOffset = txtoffset;      
      obj.redrawTextLabels(); % to utilize txtoffset
    end
    
    function refreshMarkerProps(obj)
      % Refresh marker properties (not including .Marker, .Color) on .hPts
      % based on .tfAdjusted, .lastSetAllUnadjustedResetType, .hPtsMarkerPV*
      
      resetType = obj.lastSetAllUnadjustedResetType;
      tfAdj = obj.tfAdjusted;
      
      % resetType describes/applies to those els of tfAdj that are false
      tfUnadjPredicted = ~tfAdj & ...
                         resetType==LabelCoreTemplateResetType.RESETPREDICTED;
      tfNotUnadjPredicted = ~tfUnadjPredicted;
      set(obj.hPts(tfNotUnadjPredicted),obj.hPtsMarkerPVNotPredUnadjusted);
      set(obj.hPts(tfUnadjPredicted),obj.hPtsMarkerPVPredUnadjusted);
    end

  end
  
  methods % template
    
    function tt = getTemplate(obj)
      % Create a template struct from current pts
      
      tt = struct();
      tt.pts = obj.getLabelCoords();
      lbler = obj.labeler;
      if lbler.hasTrx
        [x,y,th] = lbler.currentTargetLoc();
        tt.loc = [x y];
        tt.theta = th;
      else
        tt.loc = [nan nan];
        tt.theta = nan;
      end
    end
    
    function setTemplate(obj,tt)
      % Set "white points" to template.

      lbler = obj.labeler;      
      tfTemplateHasTarget = ~any(isnan(tt.loc)); % && ~isnan(tt.theta); 
      % For some projects (e.g. larva, theta can be nan. So shouldn't test
      % theta. MK 20250728
      tfHasTrx = lbler.hasTrx;
      
      if tfHasTrx && ~tfTemplateHasTarget
        warningNoTrace('LabelCoreTemplate:template',...
          'Using template saved without target coordinates');
      elseif ~tfHasTrx && tfTemplateHasTarget
        warningNoTrace('LabelCoreTemplate:template',...
          'Template saved with target coordinates.');
      end
        
      if tfTemplateHasTarget
        [x1,y1,th1] = lbler.currentTargetLoc;
        if isnan(th1-tt.theta)
          xys = transformPoints(tt.pts,tt.loc,0,[x1 y1],0);
        else
          xys = transformPoints(tt.pts,tt.loc,tt.theta,[x1 y1],th1);
        end
      else        
        xys = tt.pts;
      end
      
      obj.assignLabelCoords(xys,'tfClip',true);
      obj.enterAdjust(LabelCoreTemplateResetType.RESET,false);
    end
    
    function setRandomTemplate(obj)
      lbler = obj.labeler;
      [x0,y0] = lbler.currentTargetLoc('nowarn',true);
      nr = lbler.movienr;
      nc = lbler.movienc;
      r = round(max(nr,nc)/6);

      n = obj.nPts;
      x = x0 + r*2*(rand(n,1)-0.5);
      y = y0 + r*2*(rand(n,1)-0.5);
      obj.assignLabelCoords([x y],'tfClip',true);
    end
    
  end
  
  methods (Access=private)
    
    function enterAdjust(obj,resetType,tfClearLabeledPos)
      % Enter adjustment state for current frame/tgt.
      %
      % resetType: LabelCoreTemplateResetType
      % if tfClearLabeledPos, clear labeled pos.
      
      if resetType > LabelCoreTemplateResetType.NORESET
        obj.setAllPointsUnadjusted(resetType);
      end
      
      if tfClearLabeledPos
        obj.labeler.labelPosClear();
      end
        
      obj.iPtMove = nan;
      obj.tfMoved = false;
      
      set(obj.tbAccept,'BackgroundColor',[0.6,0,0],'String','Accept',...
        'Value',0,'Enable','on');
      obj.state = LabelState.ADJUST;
    end
        
    function enterAccepted(obj,tfSetLabelPos)
      % Enter accepted state for current frame/tgt. All points colored. If
      % tfSetLabelPos, all points/tags written to labelpos/labelpostag.
         
      obj.setAllPointsAdjusted();
      obj.clearSelected();
      
      if tfSetLabelPos
        obj.storeLabels();
      end
      set(obj.tbAccept,'BackgroundColor',[0,0.4,0],'String','Labeled',...
        'Value',1,'Enable','off');
      obj.state = LabelState.ACCEPTED;
    end
    
    function storeLabels(obj)
      
      xy = obj.getLabelCoords();
      obj.labeler.labelPosSet(xy);
      obj.setLabelPosTagFromEstOcc();
      
    end
    
    function setPointAdjusted(obj,iSel)
      if ~obj.tfAdjusted(iSel)
        obj.tfAdjusted(iSel) = true;
        clr = obj.ptsPlotInfo.Colors(iSel,:);
        pv = obj.hPtsMarkerPVNotPredUnadjusted;
        pv.Color = clr;
        set(obj.hPts(iSel),pv);
        if ~isempty(obj.hPtsOcc),
          set(obj.hPtsOcc(iSel),pv);
        end
        set(obj.hPtsTxt(iSel),'FontAngle','normal');
      end
    end
    
    function setAllPointsAdjusted(obj)
      clrs = obj.ptsPlotInfo.Colors;
      pv = obj.hPtsMarkerPVNotPredUnadjusted;
      for i=1:obj.nPts
        pv.Color = clrs(i,:);
        set(obj.hPts(i),pv);
        if ~isempty(obj.hPtsOcc),
          set(obj.hPtsOcc(i),pv);
        end
        set(obj.hPtsTxt(i),'FontAngle','normal');
      end
      obj.tfAdjusted(:) = true;
    end
    
    function setAllPointsUnadjusted(obj,resetType)
      assert(isa(resetType,'LabelCoreTemplateResetType'));
      switch resetType
        case LabelCoreTemplateResetType.RESET
          pv = obj.hPtsMarkerPVNotPredUnadjusted;
          pv.Color = obj.ptsPlotInfo.TemplateMode.TemplatePointColor;
          set(obj.hPts,pv);
          set(obj.hPtsOcc,pv);
          set(obj.hPtsTxt,'FontAngle','normal');
        case LabelCoreTemplateResetType.RESETPREDICTED
          %clrs = rgbbrighten(obj.ptsPlotInfo.Colors,0);
          clrs = obj.ptsPlotInfo.Colors;
          pv = obj.hPtsMarkerPVPredUnadjusted;
          for i=1:obj.nPts
            pv.Color = clrs(i,:);
            set(obj.hPts(i),pv);
            if ~isempty(obj.hPtsOcc),
              set(obj.hPtsOcc(i),pv);
            end
            set(obj.hPtsTxt(i),'FontAngle','italic');
          end
        otherwise
          assert(false);
      end
      obj.tfAdjusted(:) = false;
      obj.lastSetAllUnadjustedResetType = resetType;
    end
            
    function toggleEstOccPoint(obj,iPt)
      obj.tfEstOcc(iPt) = ~obj.tfEstOcc(iPt);
      obj.refreshPtMarkers('iPts',iPt);
      if obj.state==LabelState.ACCEPTED
        %obj.enterAdjust(LabelCoreTemplateResetType.NORESET,false);
        obj.storeLabels();
      end
    end
    
%     function refreshTxLabelCoreAux(obj)
%       iPt0 = obj.kpfIPtFor1Key;
%       iPt1 = iPt0+9;
%       str = sprintf('Hotkeys 1-9,0 map to points %d-%d, ` (backquote) toggles',iPt0,iPt1);
%       obj.txLblCoreAux.String = str;      
%     end
            
  end
  
end
