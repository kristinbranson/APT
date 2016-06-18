classdef LabelCoreErrorCorrect < LabelCore
  
  % Similar to Template mode
  %
  % On entering/navigating to a new frame:
  % - labels are assigned to pt coords. This includes NaN for missing (off-screen)
  % - A frame is either marked/locked or unmarked/unlocked. Currently all
  % pts in a frame are either marked or unmarked together.
  % -- If marked, pt display is one way, button is 'locked'.
  % -- If unmarked, pt display is another way, button is 'lock'.
  %
  % - Mousedrag adjusts pts on the fly. The position is written to Labeler
  % on WBUF (all pts in frame are locked on WBUF).
  % - <hotkey>-click or <hotkey>-arrow. Position written to Labeler, all 
  % pts in frame are locked.
  % - Unlock button unmarks all pts but they are still labeled
  % - There is currently no way to remove/clear labels
  
  properties
    iPtMove;     % scalar. Either nan, or index of pt being moved
    tfMoved;     % scalar logical; if true, pt being moved was actually moved
    
    tfAdjusted;  % nPts x 1 logical vec. If true, pt has been adjusted from original loc
    tfPtSel;     % nPts x 1 logical vec. If true, pt is currently selected
    
    kpfIPtFor1Key;  % scalar positive integer. This is the point index that 
                 % the '1' hotkey maps to, eg typically this will take the 
                 % values 1, 11, 21, ...
                 
    unmarkedLinePVs; % cell array PV pairs for unmarked lines
    markedLinePVs; % etc
  end  
  
  methods
    
    function set.kpfIPtFor1Key(obj,val)
      obj.kpfIPtFor1Key = val;
      obj.refreshTxLabelCoreAux();
    end
    
  end
  
  methods
    
    function obj = LabelCoreErrorCorrect(varargin)
      obj = obj@LabelCore(varargin{:});
    end
    
    function initHook(obj)
      npts = obj.nPts;
      obj.tfAdjusted = false(npts,1);
      obj.tfPtSel = false(npts,1);
      
      obj.txLblCoreAux.Visible = 'on';
      obj.kpfIPtFor1Key = 1;
      obj.refreshTxLabelCoreAux();
      
      ppi = obj.ptsPlotInfo;
      ppiecm = ppi.ErrorCorrectMode;
      obj.unmarkedLinePVs = {'Marker' ppi.Marker ...
        'MarkerSize' ppi.MarkerSize ...
        'LineWidth' ppi.LineWidth};
      obj.markedLinePVs = {'Marker' ppiecm.MarkedMarker ...
        'MarkerSize' ppiecm.MarkedMarkerSize ...
        'LineWidth' ppiecm.MarkedLineWidth};
    end
    
  end
  
  methods
    
    function newFrame(obj,iFrm0,iFrm1,iTgt)
      obj.newFrameAndTarget(iFrm0,iFrm1,iTgt,iTgt);
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm)
      obj.newFrameAndTarget(iFrm,iFrm,iTgt0,iTgt1);
    end
    
    function newFrameAndTarget(obj,iFrm0,iFrm1,iTgt0,iTgt1)
      [tflabeled,lpos,lpostag,lposmarked] = obj.labeler.labelPosIsLabeled(iFrm1,iTgt1);
      obj.assignLabelCoords(lpos,'lblTags',lpostag);
      tfall = all(lposmarked(:));
      tfany = any(lposmarked(:));
      assert(tfall || ~tfany,...
        'Expected all points to be either marked or unmarked in concert.');
      
      tfCurrFrameMarked = tfall;
      if tfCurrFrameMarked
        obj.enterLocked(false);
      else
        obj.enterUnlocked(false);
      end      
    end
    
    function acceptLabels(obj) 
      obj.enterLocked(true);
    end
    
    function unAcceptLabels(obj)
      obj.enterUnlocked(true);
    end 
    
    function axBDF(obj,src,evt) %#ok<INUSD>
      [tf,iSel] = obj.anyPointSelected();
      if tf
        pos = get(obj.hAx,'CurrentPoint');
        pos = pos(1,1:2);
        obj.assignLabelCoordsIRaw(pos,iSel);
        %obj.setPointAdjusted(iSel);
        obj.toggleSelectPoint(iSel);
        if obj.tfOcc(iSel)
          obj.tfOcc(iSel) = false;
          obj.refreshOccludedPts();
        end
        
        obj.enterLocked(true);
      end     
    end
    
    function ptBDF(obj,src,evt) 
      switch evt.Button
        case 1
          tf = obj.anyPointSelected();
          if tf
            % none
          else
            % prepare for click-drag of pt
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
      iPt = obj.iPtMove;
      if ~isnan(iPt)
        ax = obj.hAx;
        tmp = get(ax,'CurrentPoint');
        pos = tmp(1,1:2);
        obj.tfMoved = true;
        obj.assignLabelCoordsIRaw(pos,iPt);
        %obj.setPointAdjusted(iPt);
      end
    end
    
    function wbuf(obj,src,evt) %#ok<INUSD>
      iPt = obj.iPtMove;
      if ~isnan(iPt) 
        % point was click/dragged
        
%           obj.clearSelected(iPt);
%           obj.toggleSelectPoint(iPt);
        if obj.tfMoved
          obj.enterLocked(true);
        end
      end
      
      obj.iPtMove = nan;
      obj.tfMoved = false;
    end
    
    function kpf(obj,src,evt) %#ok<INUSL>
      key = evt.Key;
      modifier = evt.Modifier;      
      tfCtrl = any(strcmp('control',modifier));
      tfShft = any(strcmp('shift',modifier));
      
      switch key
        case {'s' 'space'}
%           if obj.state==LabelState.ADJUST
%             obj.acceptLabels();
%           end
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
            switch key
              case 'leftarrow'
                xl = xlim(obj.hAx);
                dx = diff(xl);
                if tfShift
                  xy(1) = xy(1) - dx/obj.DXFACBIG;
                else
                  xy(1) = xy(1) - dx/obj.DXFAC;
                end
                xy(1) = max(xy(1),1);
              case 'rightarrow'
                xl = xlim(obj.hAx);
                dx = diff(xl);
                if tfShift
                  xy(1) = xy(1) + dx/obj.DXFACBIG;
                else
                  xy(1) = xy(1) + dx/obj.DXFAC;
                end
                xy(1) = min(xy(1),obj.labeler.movienc);
              case 'uparrow'
                yl = ylim(obj.hAx);
                dy = diff(yl);
                if tfShift
                  xy(2) = xy(2) - dy/obj.DXFACBIG;
                else
                  xy(2) = xy(2) - dy/obj.DXFAC;
                end
                xy(2) = max(xy(2),1);
              case 'downarrow'
                yl = ylim(obj.hAx);
                dy = diff(yl);
                if tfShift
                  xy(2) = xy(2) + dy/obj.DXFACBIG;
                else
                  xy(2) = xy(2) + dy/obj.DXFAC;
                end
                xy(2) = min(xy(2),obj.labeler.movienr);
            end
            obj.assignLabelCoordsIRaw(xy,iSel);
            switch obj.state
              case LabelState.ADJUST
                obj.enterlocked(true);
              case LabelState.ACCEPTED
                % none
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
      warning('LabelCoreEC:axOcc','Occluded currently unsupported.');
%       [tf,iSel] = obj.anyPointSelected();
%       if tf
%         %obj.setPointAdjusted(iSel);
%         obj.toggleSelectPoint(iSel);
%         obj.tfOcc(iSel) = true;
%         obj.tfEstOcc(iSel) = false;
%         obj.refreshOccludedPts();
%         obj.refreshEstOccPts('iPts',iSel);
%         switch obj.state
%           case LabelState.ADJUST
%             % none
%           case LabelState.ACCEPTED
%             obj.enterUnlocked();
%         end
%       end   
    end

    function h = getLabelingHelp(obj) %#ok<MANU>
      h = { ...
        '* A/D, LEFT/RIGHT, or MINUS(-)/EQUAL(=) decrements/increments the frame shown.'
        '* <ctrl>+A/D, LEFT/RIGHT etc decrement/increment by a settable increment.'
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
      
      iPts = myparse(varargin,'iPts',1:obj.nPts);
      obj.refreshPtMarkers(iPts);
    end
        
  end
    
  methods (Access=private)
        
    function enterUnlocked(obj,tfSetLabelUnmarked)
      % Enter unlocked state for current frame/tgt.
       
      obj.iPtMove = nan;
      obj.tfMoved = false;
      
      if tfSetLabelUnmarked
        obj.labeler.labelPosSetUnmarked();
      end
      
      set(obj.hPts,obj.unmarkedLinePVs{:});
      set(obj.tbAccept,'BackgroundColor',[0,1,0],'String','Unlocked',...
        'Value',0,'Enable','on');
      obj.state = LabelState.ADJUST;
    end
        
    function enterLocked(obj,tfSetLabelPos)
      % Enter accepted state for current frame/tgt. All points colored. If
      % tfSetLabelPos, all points/tags written to labelpos/labelpostag.
      
      obj.tfAdjusted(:) = true;
      obj.clearSelected();      
      set(obj.hPts,obj.markedLinePVs{:});

      if tfSetLabelPos
        xy = obj.getLabelCoords();
        obj.labeler.labelPosSet(xy);
        obj.setLabelPosTagFromEstOcc();
      end
      set(obj.tbAccept,'BackgroundColor',[1,0,0],'String','Locked',...
        'Value',1,'Enable','on');
      obj.state = LabelState.ACCEPTED;
    end
    
%    function setPointAdjusted(obj,iSel)
      % none for now; differ from template mode
      
%       if ~obj.tfAdjusted(iSel)
%         obj.tfAdjusted(iSel) = true;
%         clr = obj.ptsPlotInfo.Colors(iSel,:);
%         set(obj.hPts(iSel),'Color',clr);
%         set(obj.hPtsOcc(iSel),'Color',clr);
%       end
%    end

    function toggleSelectPoint(obj,iPt)
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
      obj.tfEstOcc(iPt) = ~obj.tfEstOcc(iPt);
      obj.refreshEstOccPts('iPts',iPt);
      if obj.state==LabelState.ACCEPTED
        obj.enterUnlocked(false);
      end
    end
    
    function refreshPtMarkers(obj,iPts)
      % Update obj.hPts Markers based on .tfEstOcc and .tfPtSel.
      
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