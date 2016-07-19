classdef LabelCoreMultiViewTemplate < LabelCore

  % Similar to LabelCoreTemplate with multiple axes. Each axis has its own
  % template pts and you can adjust/accept as usual.
  %
  % In calRig mode, it also supports selection of axes, and 
  % - Epipolar mode. "Projecting" a selected point will show EPLs in other
  % views. Dragging a projected pt live-updates its EPLs.
  % - Stereo mode. "Projecting" a pair of pts will show EPLs and also a
  % reconstructed 3rd pt (3 dots).
  
  properties
    supportsMultiView = true;
  end
  
  properties
    iPt2Ax       % [npts]. iPt2Ax(iPt) gives the axis index for iPt
    
  
    iPtMove;     % scalar. Either nan, or index of pt being moved
    tfMoved;     % scalar logical; if true, pt being moved was actually moved
    
    tfAdjusted;  % nPts x 1 logical vec. If true, pt has been adjusted from template
    tfPtSel;     % nPts x 1 logical vec. If true, pt is currently selected
    
    kpfIPtFor1Key;  % scalar positive integer. This is the point index that 
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
    
    function obj = LabelCoreMultiViewTemplate(varargin)
      obj = obj@LabelCore(varargin{:});
    end
    
    function delete(obj)
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
      %#MVOK
      
      obj.iPt2Ax = obj.labeler.labeledposIPt2View;
     
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
        ax = obj.hAx(obj.iPt2Ax(iPt));
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
      
      npts = obj.nPts;
      obj.tfAdjusted = false(npts,1);
      obj.tfPtSel = false(npts,1);
      
      obj.txLblCoreAux.Visible = 'on';
      obj.kpfIPtFor1Key = 1;
      obj.refreshTxLabelCoreAux();
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
      %#MVOK
      
      [tflabeled,lpos,lpostag] = obj.labeler.labelPosIsLabeled(iFrm1,iTgt1);
      if tflabeled
        obj.assignLabelCoords(lpos,'lblTags',lpostag);
        obj.enterAccepted(false);
      else
        assert(iTgt0==iTgt1,'Multiple targets unsupported.');
        assert(~obj.labeler.hasTrx,'Targets are unsupported.');
        obj.enterAdjust(true,false);
      end
    end
    
    function clearLabels(obj)
      obj.clearSelected();
      obj.enterAdjust(true,true);
    end
    
    function acceptLabels(obj) 
      obj.enterAccepted(true);
    end
    
    function unAcceptLabels(obj)
      obj.enterAdjust(false,false);
    end 
    
    function axBDF(obj,src,evt) %#ok<INUSD>
      %#MVOK
      [tf,iSel] = obj.anyPointSelected();
      if tf
        assert(isscalar(iSel));
        iAx = obj.iPt2Ax(iSel);
        ax = obj.hAx(iAx);
        pos = get(ax,'CurrentPoint');
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
            obj.enterAdjust(false,false);
        end
      end     
    end
    
    function ptBDF(obj,src,evt) 
      %#MVOK
      switch evt.Button
        case 1
          tf = obj.anyPointSelected();
          if tf
            % none
          else
            % prepare for click-drag of pt
            
            if obj.state==LabelState.ACCEPTED
              obj.enterAdjust(false,false);
            end
            iPt = get(src,'UserData');
            obj.iPtMove = iPt;
            obj.tfMoved = false;
          end
        case 3
          iPt = get(src,'UserData');
          obj.toggleEstOccPoint(iPt);
      end
    end
    
    function wbmf(obj,src,evt) %#ok<INUSD>
      %#MVOK
      if obj.state==LabelState.ADJUST
        iPt = obj.iPtMove;
        if ~isnan(iPt)
          iAx = obj.iPt2Ax(iPt);
          ax = obj.hAx(iAx);
          tmp = get(ax,'CurrentPoint');
          pos = tmp(1,1:2);
          obj.tfMoved = true;
          obj.assignLabelCoordsIRaw(pos,iPt);
          obj.setPointAdjusted(iPt);
        end
      end
    end
    
    function wbuf(obj,src,evt) %#ok<INUSD>
      %#MVOK
      if obj.state==LabelState.ADJUST
        iPt = obj.iPtMove;
        if ~isnan(iPt) && ~obj.tfMoved
          % point was clicked but not moved
          obj.clearSelected(iPt);
          obj.toggleSelectPoint(iPt);
        end
        
        obj.iPtMove = nan;
        obj.tfMoved = false;
      end
    end
    
    function kpf(obj,src,evt) %#ok<INUSL>
      %#MVOK
      key = evt.Key;
      modifier = evt.Modifier;      
      tfCtrl = any(strcmp('control',modifier));
      tfShft = any(strcmp('shift',modifier));
      
      switch key
        case {'h'}
          if tfCtrl
            obj.labelsHideToggle();
          end
        case {'s' 'space'}
          if obj.state==LabelState.ADJUST
            obj.acceptLabels();
          end
        case {'d' 'equal'}
          obj.labeler.frameUp(tfCtrl);
        case {'a' 'hyphen'}
          obj.labeler.frameDown(tfCtrl);
        case {'o'}
          [tfSel,iSel] = obj.anyPointSelected();
          if tfSel
            obj.toggleEstOccPoint(iSel);
          end
        case {'leftarrow' 'rightarrow' 'uparrow' 'downarrow'}
          [tfSel,iSel] = obj.anyPointSelected();
          if tfSel && ~obj.tfOcc(iSel)
            tfShift = any(strcmp('shift',modifier));
            xy = obj.getLabelCoordsI(iSel);
            iAx = obj.iPt2Ax(iSel);
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
                xy(1) = max(xy(1),1);
              case 'rightarrow'
                xl = xlim(ax);
                dx = diff(xl);
                if tfShift
                  xy(1) = xy(1) + dx/obj.DXFACBIG;
                else
                  xy(1) = xy(1) + dx/obj.DXFAC;
                end
                ncs = obj.labeler.movienc;
                xy(1) = min(xy(1),ncs(iAx));
              case 'uparrow'
                yl = ylim(ax);
                dy = diff(yl);
                if tfShift
                  xy(2) = xy(2) - dy/obj.DXFACBIG;
                else
                  xy(2) = xy(2) - dy/obj.DXFAC;
                end
                xy(2) = max(xy(2),1);
              case 'downarrow'
                yl = ylim(ax);
                dy = diff(yl);
                if tfShift
                  xy(2) = xy(2) + dy/obj.DXFACBIG;
                else
                  xy(2) = xy(2) + dy/obj.DXFAC;
                end
                nrs = obj.labeler.movienr;
                xy(2) = min(xy(2),nrs(iAx));
            end
            obj.assignLabelCoordsIRaw(xy,iSel);
            switch obj.state
              case LabelState.ADJUST
                obj.setPointAdjusted(iSel);
              case LabelState.ACCEPTED
                obj.enterAdjust(false,false);
            end
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
          obj.clearSelected(iPt);
          obj.toggleSelectPoint(iPt);
      end      
    end
    
    function axOccBDF(obj,src,evt) %#ok<INUSD>
      assert(false,'Unsupported for multiview labeling');
      [tf,iSel] = obj.anyPointSelected();
      if tf
        obj.setPointAdjusted(iSel);
        obj.toggleSelectPoint(iSel);
        obj.tfOcc(iSel) = true;
        obj.tfEstOcc(iSel) = false;
        obj.refreshOccludedPts();
        obj.refreshEstOccPts('iPts',iSel);
        switch obj.state
          case LabelState.ADJUST
            % none
          case LabelState.ACCEPTED
            obj.enterAdjust(false,false);
        end
      end   
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
      obj.refreshPtMarkers(iPts);
    end
        
  end
  
  methods % template
    
    function setTemplate(obj,tt)
      % none
    end
    
    function tt = getTemplate(obj) %#ok<MANU>
      tt = [];
    end
    
    function setRandomTemplate(obj)
      %# MVOK
      
      lbler = obj.labeler;
      mrs = lbler.movieReader;
      movszs = [[mrs.nr]' [mrs.nc]']; % [nview x 2]. col1: nr. col2: nc
      
      xy = nan(obj.nPts,2);
      for iPt=1:obj.nPts
        iAx = obj.iPt2Ax(iPt);
        nr = movszs(iAx,1);
        nc = movszs(iAx,2);
        xy(iPt,1) = nc/2 + nc/3*2*(rand-0.5);
        xy(iPt,2) = nr/2 + nr/3*2*(rand-0.5);
      end
      obj.assignLabelCoords(xy,'tfClip',false);        
    end
    
  end
  
  methods (Access=private)
    
    function enterAdjust(obj,tfResetPts,tfClearLabeledPos)
      % Enter adjustment state for current frame/tgt.
      %
      % if tfReset, reset all points to pre-adjustment (white).
      % if tfClearLabeledPos, clear labeled pos.
      
      %#MVOK
      
      if tfResetPts
        tpClr = obj.ptsPlotInfo.TemplateMode.TemplatePointColor;
        arrayfun(@(x)set(x,'Color',tpClr),obj.hPts);
        arrayfun(@(x)set(x,'Color',tpClr),obj.hPtsOcc);
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
      
      %#MVOK
            
      nPts = obj.nPts;
      ptsH = obj.hPts;
      clrs = obj.ptsPlotInfo.Colors;
      for i = 1:nPts
        set(ptsH(i),'Color',clrs(i,:));
        set(obj.hPtsOcc(i),'Color',clrs(i,:));
      end
      
      obj.tfAdjusted(:) = true;
      obj.clearSelected();
      
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
      %#MVOK
      if ~obj.tfAdjusted(iSel)
        obj.tfAdjusted(iSel) = true;
        clr = obj.ptsPlotInfo.Colors(iSel,:);
        set(obj.hPts(iSel),'Color',clr);
        set(obj.hPtsOcc(iSel),'Color',clr);
      end
    end

    function toggleSelectPoint(obj,iPt)
      %#MVOK
      tfSel = ~obj.tfPtSel(iPt);
      obj.tfPtSel(:) = false;
      obj.tfPtSel(iPt) = tfSel;

      obj.refreshPtMarkers(iPt);
      % Also update hPtsOcc markers
      if tfSel
        mrkr = obj.ptsPlotInfo.TemplateMode.SelectedPointMarker;
      else
        mrkr = obj.ptsPlotInfo.Marker;
      end
      set(obj.hPtsOcc(iPt),'Marker',mrkr);
    end
    
    function toggleEstOccPoint(obj,iPt)
      %#MVOK
      obj.tfEstOcc(iPt) = ~obj.tfEstOcc(iPt);
      obj.refreshEstOccPts('iPts',iPt);
      if obj.state==LabelState.ACCEPTED
        obj.enterAdjust(false,false);
      end
    end
    
    function refreshPtMarkers(obj,iPts)
      % Update obj.hPts Markers based on .tfEstOcc and .tfPtSel.

      %#MVOK
      
      ppi = obj.ptsPlotInfo;
      ppitm = ppi.TemplateMode;

      hPoints = obj.hPts(iPts);
      tfSel = obj.tfPtSel(iPts);
      tfEO = obj.tfEstOcc(iPts);
      
      set(hPoints(tfSel & tfEO),'Marker',ppitm.SelectedOccludedMarker); % historical quirk, use props instead of ppi; fix this at some pt
      set(hPoints(tfSel & ~tfEO),'Marker',ppitm.SelectedPointMarker);
      set(hPoints(~tfSel & tfEO),'Marker',ppi.OccludedMarker);
      set(hPoints(~tfSel & ~tfEO),'Marker',ppi.Marker);
    end
      
    function [tf,iSelected] = anyPointSelected(obj)
      tf = any(obj.tfPtSel);
      iSelected = find(obj.tfPtSel,1);
    end
    
    function clearSelected(obj,iExclude)
      tf = obj.tfPtSel;
      if exist('iExclude','var')>0
        tf(iExclude) = false;
      end
      iSel = find(tf);
      for i = iSel(:)'
        obj.toggleSelectPoint(i);
      end
    end
    
    function refreshTxLabelCoreAux(obj)
      iPt0 = obj.kpfIPtFor1Key;
      iPt1 = iPt0+9;
      str = sprintf('Hotkeys 0-9 map to points %d-%d',iPt0,iPt1);
      obj.txLblCoreAux.String = str;      
    end
            
  end
  
end