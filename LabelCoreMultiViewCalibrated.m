classdef LabelCoreMultiViewCalibrated < LabelCore

  % Hitting a number key gets you "working on" a certain pt. Other pts
  % dimmed; working pts are highlighted in all views
 
  % Clicking on a working point will do EPL projection on all other
  % views. The working point can be dragged etc. When the working pt is
  % first clicked, it will become colored, with EPLs in other views having 
  % the same color.
  % 
  % At this pt, click-dragging (adjusting) another pt will provide 
  % 3d-reconstructed point spreads in the third/remaining views. 
  % Right-click on either of the first two points to "unadjust" it.
  %
  % When done, hit another number key to change the working point.
  %
  % When done with all points, hit Accept to Accept labels.
  %
  % This requires a 'CalibratedRig' that knows how to compute EPLs and
  % reconstruct 3dpts.

  
  properties
    supportsMultiView = true;
  end
  
  properties

    iPt2iAx       % [npts]. iPt2iAx(iPt) gives the axis index for iPt
    % .hPts       % [npts] from LabelCore. hLine handles for pts (in
    %               respective axes)
    % .hPtsTxt    % [npts]
    iSet2iPt      % [nset x nview]. A point 'set' is a nview-tuple of point 
                  % indices that represent a single physical (3d) point.
                  % .iSet2iPt(iSet,:) are the nview pt indices that
                  % correspond to pointset iSet.
  end
  properties (Dependent)
    nView         % scalar
    nPointSet     % scalar, number of 'point sets'.    
  end  
  
  %% Working set
  properties
    iSetWorking      % scalar. Set index of working set. Can be nan for no working set.
    kpfIPtFor1Key;   % scalar positive integer. This is the point index that
                     % the '1' hotkey maps to, eg typically this will take the
                     % values 1, 11, 21, ...
  end
  
  %% Projections
  properties
    pjtIPts          % [1 2] vector; current first/anchor working pt. either
                     % [nan nan], [iPt1 nan], or [iPt1 iPt2]
    pjtHLinesEpi     % [nview]. line handles for epipolar lines
    pjtHLinesRecon   % [nview]. line handles for reconstructed pts
    
    pjtCalRigFilename % full path filename for calibration object
    pjtCalRig        % Scalar some-kind-of-calibration object
  end
  properties (Dependent)
    pjtState         % either 0, 1, or 2 for number of defined working pts
  end
  
  %% Misc
  properties
    % click-drag
    iPtMove;     % scalar. Either nan, or index of pt being moved
    tfMoved;     % scalar logical; if true, pt being moved was actually moved


    tfAdjusted;  % nPts x 1 logical vec. If true, pt has been adjusted from template

%     tfPtSel;     % nPts x 1 logical vec. If true, pt is currently selected  
  end
  
  methods % dep prop getters
    function v = get.nView(obj)
      v = obj.labeler.nview;
    end
    function v = get.nPointSet(obj)
      v = size(obj.iSet2iPt,1);
    end
    function v = get.pjtState(obj)
      v = nnz(~isnan(obj.pjtIPts));
    end
  end
  
  methods 
    
    function set.kpfIPtFor1Key(obj,val)
      obj.kpfIPtFor1Key = val;
      obj.refreshTxLabelCoreAux();
    end
    
  end
  
  methods
    
    function obj = LabelCoreMultiViewCalibrated(varargin)
      obj = obj@LabelCore(varargin{:});
    end
    
    function delete(obj)
      % CALTODO
      gdata = obj.labeler.gdata;
      hFigs = gdata.figs_all;
      hFigsAddnl = setdiff(hFigs,gdata.figure);

      % Remove any pointers to this object from callbacks in hFigsAddnl.
      % Callbacks in gdata.figure (primary figure) are fine to leave as
      % they are LabelCore's responsibility.
      
      hTmp = findall(hFigsAddnl,'-property','KeyPressFcn','-not','Tag','edit_frame');
      set(hTmp,'KeyPressFcn',[]);
      set(hFigsAddnl,'WindowButtonMotionFcn',[]);
      set(hFigsAddnl,'WindowButtonUpFcn',[]);
    end
    
    function initHook(obj)
      obj.iPt2iAx = obj.labeler.labeledposIPt2View;
      obj.iSet2iPt = obj.labeler.labeledposIPtSetMap;
      
      % redefine .hPts, .hPtsTxt (originally initted in LabelCore.init())
      deleteValidHandles(obj.hPts);
      deleteValidHandles(obj.hPtsTxt);
      obj.hPts = gobjects(obj.nPts,1);
      obj.hPtsTxt = gobjects(obj.nPts,1);
      ppi = obj.ptsPlotInfo;
      for iPt=1:obj.nPts
        ptsArgs = {nan,nan,ppi.Marker,...
          'MarkerSize',ppi.MarkerSize,...
          'LineWidth',ppi.LineWidth,...
          'Color',ppi.Colors(iPt,:),...
          'UserData',iPt,...
          'HitTest','on',...
          'ButtonDownFcn',@(s,e)obj.ptBDF(s,e)};
        ax = obj.hAx(obj.iPt2iAx(iPt));
        obj.hPts(iPt) = plot(ax,ptsArgs{:});
        obj.hPtsTxt(iPt) = text(nan,nan,num2str(iPt),...
          'Parent',ax,...
          'Color',ppi.Colors(iPt,:),...
          'FontSize',ppi.FontSize,...
          'Hittest','off');
      end

      % set callbacks for addnl figs/axes
      gdata = obj.labeler.gdata;
      hFigs = gdata.figs_all;
      hFigsAddnl = setdiff(hFigs,gdata.figure);
      hTmp = findall(hFigsAddnl,'-property','KeyPressFcn','-not','Tag','edit_frame');
      set(hTmp,'KeyPressFcn',@(s,e)obj.kpf(s,e)); % main axis KPF set in LabelCore.init()
      set(hFigsAddnl,'WindowButtonMotionFcn',@(s,e)obj.wbmf(s,e));
      set(hFigsAddnl,'WindowButtonUpFcn',@(s,e)obj.wbuf(s,e));
      
      obj.setRandomTemplate();
      
%       npts = obj.nPts;
      
      obj.tfAdjusted = false(obj.nPts,1);
%       obj.tfPtSel = false(npts,1);
%       
      obj.txLblCoreAux.Visible = 'on';
      obj.kpfIPtFor1Key = 1;
%       obj.refreshTxLabelCoreAux();

      obj.workingSetClear();
      obj.projectionInit();
    end
    
  end
  
  methods

    % newFrameAndTarget() combines all the brains of transitions for 
    % convenience reasons
    
    function newFrame(obj,iFrm0,iFrm1,iTgt)
      obj.newFrameAndTarget(iFrm0,iFrm1,iTgt,iTgt);
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm)
      obj.newFrameAndTarget(iFrm,iFrm,iTgt0,iTgt1);
    end
    
    function newFrameAndTarget(obj,iFrm0,iFrm1,iTgt0,iTgt1)
      %#CALOK
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
      
      % projection state: very crude refresh
      obj.projectionRefresh();
    end
    
    function clearLabels(obj)
      %#CALOK
      obj.enterAdjust(true,true);
      obj.workingSetClear();
      obj.projectionClear();
    end
    
    function acceptLabels(obj)
      obj.enterAccepted(true);
    end
    
    function unAcceptLabels(obj)
      obj.enterAdjust(false,false);
    end 
    
    function axBDF(obj,src,evt) %#ok<INUSD>
    end
    
    function ptBDF(obj,src,evt)
      %#CALOK
      switch evt.Button
        case 1
          iPt = src.UserData;
          tfInWS = obj.workingSetPointInWS(iPt);
          if tfInWS            
            % prepare for click-drag of pt
            obj.iPtMove = iPt;
            obj.tfMoved = false;
          end
        case 3
          % none
%           iPt = get(src,'UserData');
%           obj.toggleEstOccPoint(iPt);
      end
    end
    
    function wbmf(obj,src,evt) %#ok<INUSD>
      %#CALOK
      
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
      %#CALOK
      
      iPt = obj.iPtMove;
      if ~isnan(iPt)
        if obj.tfMoved
          % none
        else
          % point was clicked but not moved
          obj.projectToggleState(iPt);
        end
      end
      obj.iPtMove = nan;
      obj.tfMoved = false;
    end
    
    function kpf(obj,src,evt) %#ok<INUSL>
      %#CALOK
      key = evt.Key;
      modifier = evt.Modifier;      
      tfCtrl = any(strcmp('control',modifier));
      tfShft = any(strcmp('shift',modifier));
      
      switch key
        case {'s' 'space'}
          if obj.state==LabelState.ADJUST
            obj.acceptLabels();
          end
        case {'d' 'equal'}
          obj.labeler.frameUp(tfCtrl);
        case {'a' 'hyphen'}
          obj.labeler.frameDown(tfCtrl);
        case {'o'}
%           [tfSel,iSel] = obj.anyPointSelected();
%           if tfSel
%             obj.toggleEstOccPoint(iSel);
%           end
        case {'leftarrow' 'rightarrow' 'uparrow' 'downarrow'}
          %[tfSel,iSel] = obj.anyPointSelected();
          if false % tfSel && ~obj.tfOcc(iSel)
%             tfShift = any(strcmp('shift',modifier));
%             xy = obj.getLabelCoordsI(iSel);
%             iAx = obj.iPt2iAx(iSel);
%             ax = obj.hAx(iAx);
%             switch key
%               case 'leftarrow'
%                 xl = xlim(ax);
%                 dx = diff(xl);
%                 if tfShift
%                   xy(1) = xy(1) - dx/obj.DXFACBIG;
%                 else
%                   xy(1) = xy(1) - dx/obj.DXFAC;
%                 end
%                 xy(1) = max(xy(1),1);
%               case 'rightarrow'
%                 xl = xlim(ax);
%                 dx = diff(xl);
%                 if tfShift
%                   xy(1) = xy(1) + dx/obj.DXFACBIG;
%                 else
%                   xy(1) = xy(1) + dx/obj.DXFAC;
%                 end
%                 ncs = obj.labeler.movienc;
%                 xy(1) = min(xy(1),ncs(iAx));
%               case 'uparrow'
%                 yl = ylim(ax);
%                 dy = diff(yl);
%                 if tfShift
%                   xy(2) = xy(2) - dy/obj.DXFACBIG;
%                 else
%                   xy(2) = xy(2) - dy/obj.DXFAC;
%                 end
%                 xy(2) = max(xy(2),1);
%               case 'downarrow'
%                 yl = ylim(ax);
%                 dy = diff(yl);
%                 if tfShift
%                   xy(2) = xy(2) + dy/obj.DXFACBIG;
%                 else
%                   xy(2) = xy(2) + dy/obj.DXFAC;
%                 end
%                 nrs = obj.labeler.movienr;
%                 xy(2) = min(xy(2),nrs(iAx));
%             end
%             obj.assignLabelCoordsIRaw(xy,iSel);
%             switch obj.state
%               case LabelState.ADJUST
%                 obj.setPointAdjusted(iSel);
%               case LabelState.ACCEPTED
%                 obj.enterAdjust(false,false);
%             end
          elseif strcmp(key,'leftarrow')
            if tfShft
              obj.labeler.frameUpNextLbled(true);
            else
              obj.labeler.frameDown(tfCtrl);
            end
          elseif strcmp(key,'rightarrow')
            if tfShft
              obj.labeler.frameUpNextLbled(false);
            else
              obj.labeler.frameUp(tfCtrl);
            end
          end
        case {'backquote'}
          iPt = obj.kpfIPtFor1Key+10;
          if iPt > obj.nPts
            iPt = 1;
          end
          obj.kpfIPtFor1Key = iPt;
        case {'0' '1' '2' '3' '4' '5' '6' '7' '8' '9'}
          iPt = str2double(key);
          if iPt==0
            iPt = 10;
          end
          iPt = iPt+obj.kpfIPtFor1Key-1;
          if iPt > obj.nPts
            return;
          end
          obj.workingSetToggle(iPt);
      end      
    end
    
    function axOccBDF(obj,src,evt) %#ok<INUSD>
      assert(false,'Unsupported for multiview labeling');     
    end

    function h = getLabelingHelp(obj) %#ok<MANU>
      h = { ...
        '* A/D, LEFT/RIGHT, or MINUS(-)/EQUAL(=) decrements/increments the frame shown.'
        '* <ctrl>+A/D, LEFT/RIGHT etc decrement/increment by 10 frames.'
        '* S or <space> accepts the labels for the current frame/target.'
        '* (The letter) O toggles occluded-estimated status.'
        '* 0..9 selects/unselects a point. When a point is selected:'
        '* ` (backquote) increments the mapping of the 0-9 hotkeys.'
        '* LEFT/RIGHT/UP/DOWN adjusts the point.'
        '* Shift-LEFT, etc adjusts the point by larger steps.' 
        '* Clicking on the image moves the selected point to that location.'};
    end
    
    function refreshEstOccPts(obj,varargin)
      % React to an updated .tfEstOcc.
      %
      % optional PVs
      % iPts. Defaults to 1:obj.nPts.
      
      %#MVOK
      
      iPts = myparse(varargin,'iPts',1:obj.nPts);
      %obj.refreshPtMarkers(iPts);
    end
        
  end
  
  methods % template
        
    function setRandomTemplate(obj)
      %# CALOK
      
      lbler = obj.labeler;
      mrs = lbler.movieReader;
      movszs = [[mrs.nr]' [mrs.nc]']; % [nview x 2]. col1: nr. col2: nc
      
      xy = nan(obj.nPts,2);
      for iPt=1:obj.nPts
        iAx = obj.iPt2iAx(iPt);
        nr = movszs(iAx,1);
        nc = movszs(iAx,2);
        xy(iPt,1) = nc/2 + nc/3*2*(rand-0.5);
        xy(iPt,2) = nr/2 + nr/3*2*(rand-0.5);
      end
      obj.assignLabelCoords(xy,'tfClip',false);        
    end
    
  end
  
  methods 
    
    function workingSetClear(obj)
      clrs = obj.ptsPlotInfo.Colors;
      h = obj.hPts;
      for i=1:obj.nPts
        set(h(i),'Color',clrs(i,:));
        %set(obj.hPtsTxt,'FaceAlpha',1);
      end
      obj.iSetWorking = nan;
    end
    
    function workingSetSet(obj,iSet)
      iPtsSet = obj.iSet2iPt(iSet,:);
      %iPtsComp = setdiff(1:obj.nPts,iPtsSet);

      clrs = obj.ptsPlotInfo.Colors;
      h = obj.hPts;
      for i=1:obj.nPts
        if ~any(i==iPtsSet)
          set(h(i),'Color',clrs(i,:)/2);
        end
      end
      obj.iSetWorking = iSet;
    end
    
    function workingSetToggle(obj,iSet)
      if isnan(obj.iSetWorking)
        obj.workingSetSet(iSet);
      else
        tfMatch = obj.iSetWorking==iSet;
        obj.workingSetClear();
        if ~tfMatch
          obj.workingSetSet(iSet);
        end
      end
    end
    
    function tf = workingSetPointInWS(obj,iPt)
      % Returns true if iPt is in current working set.
      iSet = obj.iSetWorking;
      tf = ~isnan(iSet) && any(iPt==obj.iSet2iPt(iSet,:));
    end
    
    function projectionInit(obj)
      obj.pjtIPts = [nan nan];
      hLEpi = gobjects(1,obj.nView);
      hLRcn = gobjects(1,obj.nView);      
      for iV = 1:obj.nView
        ax = obj.labeler.gdata.axes_all(iV);        
        hLEpi(iV) = plot(ax,nan,nan,'-','LineWidth',2); % XXXPREF
        hLRcn(iV) = plot(ax,nan,nan,'o');
      end
      obj.pjtHLinesEpi = hLEpi;
      obj.pjtHLinesRecon = hLRcn;
    end
    
    function projectionClear(obj)
      % Clear all projection points.
      
      %# CALOK
      
      set(obj.hPts,'Marker','o'); % XXXPREF
      obj.pjtIPts = [nan nan];
      set(obj.pjtHLinesEpi,'Visible','off');
      set(obj.pjtHLinesRecon,'Visible','off');
    end
    
    function projectToggleState(obj,iPt)
      % Toggle projection status of point iPt.
      
      %#CALOK
      
      switch obj.pjtState
        case 0
          obj.projectionSetAnchor(iPt);
        case 1
          if iPt==obj.pjtIPts(1)
            obj.projectionClear();
          else
            obj.projectionSet2nd(iPt);
          end
        case 2
          idx = find(obj.pjtIPts==iPt);
          if isempty(idx)
            % none
          else
            idxOther = mod(idx,2)+1;
            iPtOther = obj.pjtIPts(idxOther);
            obj.projectionClear();
            obj.projectionSetAnchor(iPtOther);
          end
      end
    end

    function projectionSetAnchor(obj,iPt1)
      %# CALOK
            
      if ~isnan(obj.pjtIPts(1))
        obj.projectionClear();
      end
      hPt1 = obj.hPts(iPt1);
      set(hPt1,'Marker','s'); % XXXPREF
      obj.pjtIPts(1) = iPt1;
      assert(isnan(obj.pjtIPts(2)));
      
      obj.projectionRefreshEPlines();
    end
    
    function projectionRefreshEPlines(obj)
      % update EPlines based on .pjtIPt1 and coords of that hPt.
      
      assert(obj.pjtState==1);
      
      iPt1 = obj.pjtIPts(1);
      hPt1 = obj.hPts(iPt1);
      xy1 = [hPt1.XData hPt1.YData];
      iAx1 = obj.iPt2iAx(iPt1);
      iAxOther = setdiff(1:obj.nView,iAx1);
      crig = obj.pjtCalRig;
      for iAx = iAxOther
        [x,y] = crig.computeEpiPolarLine(iAx1,xy1,iAx);
        hEpi = obj.pjtHLinesEpi(iAx);
        set(hEpi,'XData',x,'YData',y,'Visible','on','Color',hPt1.Color);
      end
      set(obj.pjtHLinesEpi(iAx1),'Visible','off');
    end
    
    function projectionSet2nd(obj,iPt2)
      %# CALOK
    
      assert(~isnan(obj.pjtIPts(1)));
      assert(isnan(obj.pjtIPts(2)));
      assert(iPt2~=obj.pjtIPts(1),'Second projection point must differ from anchor point.');
      obj.pjtIPts(2) = iPt2;
      set(obj.pjtHLinesEpi,'Visible','off');
      
      obj.hPts(iPt2).Marker = 's'; %XXXPREF
      
      obj.projectionRefreshReconPts();
    end
    
    function projectionRefreshReconPts(obj)
      % update recon pts based on .pjtIPt1 and .pjtIPt2 and coords of 
      % corresponding hPts.
      
      assert(obj.pjtState==2);
      
      iPt1 = obj.pjtIPts(1);
      iPt2 = obj.pjtIPts(2);
      iAx1 = obj.iPt2iAx(iPt1);
      iAx2 = obj.iPt2iAx(iPt2);
      hPt1 = obj.hPts(iPt1);
      hPt2 = obj.hPts(iPt2);
      
      xy1 = [hPt1.XData hPt1.YData];
      xy2 = [hPt2.XData hPt2.YData];
      iAxOther = setdiff(1:obj.nView,[iAx1 iAx2]);
      crig = obj.pjtCalRig;
      for iAx = iAxOther
        [x,y] = crig.reconstruct(iAx1,xy1,iAx2,xy2,iAx);
        set(obj.pjtHLinesRecon(iAx),...
          'XData',x,'YData',y,...
          'Visible','on','Color',hPt1.Color);
      end
    end
    
    function projectionRefresh(obj)
      switch obj.pjtState
        case 0
          % none
        case 1
          obj.projectionRefreshEPlines();
        case 2
          obj.projectionRefreshReconPts();
        otherwise
          assert(false);
      end
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
    % WORKING SET
    % - When a pointset is working, then all its points are shown in full
    % alpha while others are faded.
    % - You can have no working pointsets.
    %
    % PROJECTION POINTS
    % - You can click on working pts to set them as projection pts. First
    % pt => anchor. This changes the marker to a square and creates EPLs in
    % the other views.
    % - Second pt => 2nd pt. This changes the marker and creates
    % 3dprojspreads in remaining views.
    % - Regardless of projection-ness you can click-drag any points in the
    % working set around.
    % - Single-click to release/remove a pt from anchor-ness or 2nd-ness
    % - Regardless of projection state, you can Accept, which writes
    % current positions to Labeler.
    
    function enterAdjust(obj,tfResetPts,tfClearLabeledPos)
      % Enter adjustment state for current frame/tgt.
      %
      % if tfReset, reset all points to pre-adjustment (white).
      % if tfClearLabeledPos, clear labeled pos.

      %#CALOK
      
      if tfResetPts
        tpClr = obj.ptsPlotInfo.TemplateMode.TemplatePointColor;
        arrayfun(@(x)set(x,'Color',tpClr),obj.hPts);
        obj.tfAdjusted(:) = false;
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
      
      %#CALOK
            
      nPts = obj.nPts;
      ptsH = obj.hPts;
      clrs = obj.ptsPlotInfo.Colors;
      for i = 1:nPts
        set(ptsH(i),'Color',clrs(i,:));
        set(obj.hPtsOcc(i),'Color',clrs(i,:));
      end
      
      obj.tfAdjusted(:) = true;
      
      if tfSetLabelPos
        xy = obj.getLabelCoords();
        obj.labeler.labelPosSet(xy);
        obj.setLabelPosTagFromEstOcc();
      end
      set(obj.tbAccept,'BackgroundColor',[0,0.4,0],'String','Accepted',...
        'Value',1,'Enable','on');
      obj.state = LabelState.ACCEPTED;
    end
    
    function setPointAdjusted(obj,iSel)
      %#CALOK
      if ~obj.tfAdjusted(iSel)
        obj.tfAdjusted(iSel) = true;
        clr = obj.ptsPlotInfo.Colors(iSel,:);
        set(obj.hPts(iSel),'Color',clr);
        set(obj.hPtsOcc(iSel),'Color',clr);
      end
    end
    
    function toggleEstOccPoint(obj,iPt)
      %#MVOK
      obj.tfEstOcc(iPt) = ~obj.tfEstOcc(iPt);
      obj.refreshEstOccPts('iPts',iPt);
      if obj.state==LabelState.ACCEPTED
        obj.enterAdjust(false,false);
      end
    end
    
%     function refreshPtMarkers(obj,iPts)
%       % Update obj.hPts Markers based on .tfEstOcc and .tfPtSel.
% 
%       %#MVOK
%       
%       ppi = obj.ptsPlotInfo;
%       ppitm = ppi.TemplateMode;
% 
%       hPoints = obj.hPts(iPts);
%       tfSel = obj.tfPtSel(iPts);
%       tfEO = obj.tfEstOcc(iPts);
%       
%       set(hPoints(tfSel & tfEO),'Marker',ppitm.SelectedOccludedMarker); % historical quirk, use props instead of ppi; fix this at some pt
%       set(hPoints(tfSel & ~tfEO),'Marker',ppitm.SelectedPointMarker);
%       set(hPoints(~tfSel & tfEO),'Marker',ppi.OccludedMarker);
%       set(hPoints(~tfSel & ~tfEO),'Marker',ppi.Marker);
%     end
      
%     function [tf,iSelected] = anyPointSelected(obj)
%       tf = any(obj.tfPtSel);
%       iSelected = find(obj.tfPtSel,1);
%     end
     
%     function clearSelected(obj,iExclude)
%       tf = obj.tfPtSel;
%       if exist('iExclude','var')>0
%         tf(iExclude) = false;
%       end
%       iSel = find(tf);
%       for i = iSel(:)'
%         obj.toggleSelectPoint(i);
%       end
%     end
    
    function refreshTxLabelCoreAux(obj)
      iPt0 = obj.kpfIPtFor1Key;
      iPt1 = iPt0+9;
      str = sprintf('Hotkeys 0-9 map to points %d-%d',iPt0,iPt1);
      obj.txLblCoreAux.String = str;      
    end
            
  end
  
end