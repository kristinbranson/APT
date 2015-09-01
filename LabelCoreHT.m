classdef LabelCoreHT < LabelCore
  % High-throughput Labeling
  %
  % User goes through the movie, labeling first point (1 out of Npts) in 
  % every Nframeskip'th frame. There is a single white point on the image 
  % for the last-clicked point. At each frame, clicking on the image moves 
  % the white point to the selected location and colorizes it, indicating 
  % that a location has been specified (for the current point). The movie
  % auto-advances by Nframeskip frames.
  %
  % When the end of the movie is reached, a dialog is given and labeling 
  % starts over at the beginning of the movie (for the current target), for
  % the next point. Labeling is fully complete when all points are labeled 
  % for all frames (for a given target).
  % 
  % STATE
  % The state of labeling at any time is given by:
  % 1. current frame
  % 3. current point index (iPoint = 1..Npts)
  % 4. Whether current point is unclicked (white) or clicked (color)
  % 5. current labels (.labeledpos array)
  %
  % Actions:
  %
  % - The current frame is shown with all previously labeled ptindices
  % besides iPoint in gray. The current point iPoint is either white
  % (unclicked) or colored (clicked).
  %
  % - clicking the image:
  % -- moves the pt to the clicked location (regardless of whether it was 
  %    previously clicked) 
  % -- converts white pt to colored 
  % -- writes to labels
  % -- increments frame by Nframeskip
  % --- if the movie end is hit, a dialog is shown to continue to increment
  %     iPoint.
  % 
  % - Manually navigating forward in frames does what you'd expect; if the
  % frame is labeled, the pt is shown in white; otherwise, colored.
  %
  % - Manually navigating backward is the same. Note regardless of fwd or
  % backward, other iPoints are not adjustable.
  %
  % - pbClear is always enabled:
  % -- converts colored pt to white
  % -- clears labels
  %
  % - pbOccluded: 
  % -- moves pt to upper-left and colorizes, otherwise same as clicking
  %
  % TODO
  % - For now there is no notion of targets in this node; you are labeling 
  %   one target at a time for an entire movie. If you want to label 
  %   NTarget targets, just run repeat the entire procedure NTarget times.
  % - Review Mode? -- Could just feed results through template mode?
  % - Automatic capability to "resume where you left off"?

  
  properties
    iPoint;    % scalar. Either nan, or index of pt currently being labeled
    tfClicked; % scalar logical; if true, pt was labeled

    nFrameSkip;
    unlabeledPointColor = [1 1 1];
    otherlabeledPointDarkenFac = 0.5;
  end  
  
  methods
    
    function obj = LabelCoreHT(varargin)
      obj = obj@LabelCore(varargin{:});
    end
    
    function initHook(obj)
      obj.iPoint = 1;
      obj.setRandomIPt();
      obj.tfClicked = false;

      ppi = obj.ptsPlotInfo;
      htm = ppi.HighThroughputMode;
      obj.nFrameSkip = htm.NFrameSkip;
      obj.unlabeledPointColor = htm.UnlabeledPointColor;      
    end
    
  end
  
  methods
    
    function newFrame(obj,~,iFrm1,iTgt)
      xy = obj.labeler.labeledpos(:,:,iFrm1,iTgt);
      tfLbled = ~isnan(xy(:,1)) & ~isinf(xy(:,1));
      tfOcced = isinf(xy(:,1)); 
      tfUnlbled = isnan(xy(:,1));

      iPt = obj.iPoint;
      hPoints = obj.hPts;
      hPointsTxt = obj.hPtsTxt;
      colors = obj.ptsPlotInfo.Colors;
      
      % COLORING
      % - all labeled, occluded that are not iPoint are colored but dimmed
      % - iPoint is colored if labeled/occluded, otherwise
      % unlabeledPointColor
      % - all other points (unlabeled, unoccluded, not iPoint) will be
      % hidden so coloring irrelevant
      
      tfOL = tfLbled; % other-labeled
      tfOL(iPt) = false;
      iPtOL = find(tfOL);
      darkenFac = obj.otherlabeledPointDarkenFac;
      for i = iPtOL(:)'
        clr = darkenFac*colors(i,:);
        set(hPoints(i),'Color',clr);
        % leave hPtsTxtOL color
      end
      if tfLbled(iPt)
        set(hPoints(iPt),'Color',colors(iPt,:));
      else
        set(hPoints(iPt),'Color',obj.unlabeledPointColor);
      end
      
      % POSITIONING
      % - all labeled pts are positioned per labels (including iPoint)
      % - all occed pts are positioned in corner per occluded
      % - if nonlabeled, iPoint position unchanged
      % - other nonlabeled, nonocced are hidden
      LabelCore.assignCoords2Pts(xy(tfLbled,:),hPoints(tfLbled),hPointsTxt(tfLbled));
      obj.dispOccludedPts(tfOcced);
      tfUnlbledNotIPt = tfUnlbled;
      tfUnlbledNotIPt(iPt) = false;
      LabelCore.assignCoords2Pts(nan(nnz(tfUnlbledNotIPt),2),hPoints(tfUnlbledNotIPt),hPointsTxt(tfUnlbledNotIPt));

      obj.tfClicked = tfLbled(iPt);
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm) %#ok<INUSD>
      % none
    end
    
    function clearLabels(obj)
      iPt = obj.iPoint;
      set(obj.hPts(iPt),'Color',obj.unlabeledPointColor);
      obj.tfClicked = false;
      obj.labeler.labelPosClearI(iPt);
    end
    
    function acceptLabels(obj) %#ok<MANU>
      assert(false);
    end
    
    function unAcceptLabels(obj) %#ok<MANU>
      assert(false);
    end 
    
    function axBDF(obj,src,evt) %#ok<INUSD>
      pos = get(obj.hAx,'CurrentPoint');
      pos = pos(1,1:2);
      iPt = obj.iPoint;
      obj.assignLabelCoordsI(pos,iPt);

      set(obj.hPts(iPt),'Color',obj.ptsPlotInfo.Colors(iPt,:));
      obj.labeler.labelPosSetI(pos,iPt);
      obj.tfClicked = true;
      
      obj.labeler.frameUpDF(obj.nFrameSkip);
    end
    
    function ptBDF(obj,src,evt) %#ok<INUSD>
      % none
    end
    
    function wbmf(obj,src,evt) %#ok<INUSD>
      % none
    end
    
    function wbuf(obj,src,evt) %#ok<INUSD>
      %disp('foo');
      % none
    end
    
    function pnlBDF(obj,src,evt)
      iPt = obj.iPoint;
      obj.labeler.labelPosSetOccludedI(iPt);
      tfOcc = obj.labeler.labelPosIsOccluded();
      
      obj.dispOccludedPts(tfOcc);

      obj.tfClicked = true;
      obj.labeler.frameUpDF(obj.nFrameSkip);
    end
    
    function kpf(obj,src,evt) %#ok<INUSL>
      fprintf(2,'TODO');
      key = evt.Key;
      modifier = evt.Modifier;      
      tfCtrl = any(strcmp('control',modifier));
      
      switch key
        case {'s' 'space'}
          if obj.state==LabelState.ADJUST
            obj.acceptLabels();
          end
        case {'d' 'equal'}
          obj.labeler.frameUp(tfCtrl);
        case {'a' 'hyphen'}
          obj.labeler.frameDown(tfCtrl);
        case {'leftarrow' 'rightarrow' 'uparrow' 'downarrow'}
          [tfSel,iSel] = obj.anyPointSelected();
          if tfSel
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
            obj.assignLabelCoordsI(xy,iSel);
            switch obj.state
              case LabelState.ADJUST
                obj.setPointAdjusted(iSel);
              case LabelState.ACCEPTED
                obj.enterAdjust(false,false);
            end
          elseif strcmp(key,'leftarrow')
            obj.labeler.frameDown(tfCtrl);
          elseif strcmp(key,'rightarrow')
            obj.labeler.frameUp(tfCtrl);
          end
        case {'0' '1' '2' '3' '4' '5' '6' '7' '8' '9'}
          iPt = str2double(key);
          if iPt==0
            iPt = 10;
          end
          if iPt > obj.nPts
            return;
          end
          obj.clearSelected(iPt);
          obj.toggleSelectPoint(iPt);
      end      
    end
    
    function h = getKeyboardShortcutsHelp(obj) %#ok<MANU>
      h = { ...
        '* A/D, LEFT/RIGHT, or MINUS(-)/EQUAL(=) decrements/increments the frame shown.'
        '* <ctrl>+A, <ctrl>+D, etc decrement and increment by 10 frames.'
        '* S or <space> accepts the labels for the current frame/target.'
        '* 0..9 selects/unselects a point. When a point is selected:'
        '*   LEFT/RIGHT/UP/DOWN adjusts the point.'
        '*   Shift-LEFT, etc adjusts the point by larger steps.' 
        '*   Clicking on the image moves the selected point to that location.'};
    end
    
  end
  
  methods (Access=private)   
    
    function setRandomIPt(obj)
      lbler = obj.labeler;
      [x0,y0] = lbler.currentTargetLoc();
      
      n = obj.nPts;
      xy = nan(n,2);
      xy(obj.iPoint,:) = [x0 y0];
      % AL20150901: if tfclip=true, nans get clipped to 1 or nr/nc!
      % optional arg to max() and min() instroduced in R2015b to specify
      % NaN treatment.
      obj.assignLabelCoords(xy,false); 
    end    
           
  end
  
end