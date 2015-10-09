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
    otherLabeledPointColor = [0.4 0.4 0.4];
  end  
  
  methods
    
    function obj = LabelCoreHT(varargin)
      obj = obj@LabelCore(varargin{:});
    end
    
    function initHook(obj)
      obj.iPoint = 1;
     
      % hide all pts
      n = obj.nPts;
      xy = nan(n,2);
      % AL20150901: if tfclip=true, nans get clipped to 1 or nr/nc!
      % optional arg to max() and min() introduced in R2015b to specify
      % NaN treatment.
      obj.assignLabelCoords(xy);

      obj.tfClicked = false;

      ppi = obj.ptsPlotInfo;
      htm = ppi.HighThroughputMode;
      obj.nFrameSkip = htm.NFrameSkip;
      obj.unlabeledPointColor = htm.UnlabeledPointColor;
      obj.otherLabeledPointColor = htm.OtherLabeledPointColor;
      
      set(obj.hPts,'HitTest','off');
      set(obj.hPtsTxt,'HitTest','off');
      set(obj.labeler.gdata.tbAccept,'Enable','off');
      obj.labeler.currImHud.setReadoutFields('hasLblPt',true);

      obj.setIPoint(1);
    end
    
  end
  
  methods
    
    function newFrame(obj,~,iFrm1,iTgt)
      lpos = obj.labeler.labeledposCurrMovie;
      xy = lpos(:,:,iFrm1,iTgt);
      tfUnlbled = isnan(xy(:,1));
      tfLbledOrOcc = ~tfUnlbled;

      iPt = obj.iPoint;
      hPoints = obj.hPts;
      hPointsOcc = obj.hPtsOcc;
%       hPointsTxt = obj.hPtsTxt;
%       hPointsTxtOcc = obj.hPtsTxtOcc;
      colors = obj.ptsPlotInfo.Colors;      
      
      % POSITIONING
      % - all labeled pts are positioned per labels (including iPoint)
      % - all occed pts are positioned per usual
      % - if nonlabeled, iPoint position unchanged
      % - other nonlabeled, nonocced are hidden
      if tfUnlbled(iPt)
        xy(iPt,:) = obj.getLabelCoordsI(iPt);
      end
      obj.assignLabelCoords(xy);

      % COLORING
      % - all labeled/occluded that are not iPoint are colored but dimmed
      % - iPoint is colored if labeled/occluded, otherwise
      % unlabeledPointColor
      % - all other points (unlabeled, unoccluded, not iPoint) will be
      % hidden so coloring irrelevant
      
      tfOL = tfLbledOrOcc; % other-labeled
      tfOL(iPt) = false;
      clr = obj.otherLabeledPointColor;
      set(hPoints(tfOL),'Color',clr);
      set(hPointsOcc(tfOL),'Color',clr);

      if tfLbledOrOcc(iPt)
        clr = colors(iPt,:);
      else
        clr = obj.unlabeledPointColor;
      end
      set(hPoints(iPt),'Color',clr);
      set(hPointsOcc(iPt),'Color',clr);
      
      obj.tfClicked = tfLbledOrOcc(iPt);
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm) %#ok<INUSD>
      % none
    end
    
    function clearLabels(obj)
      iPt = obj.iPoint;
      set(obj.hPts(iPt),'Color',obj.unlabeledPointColor);
      set(obj.hPtsOcc(iPt),'Color',obj.unlabeledPointColor);
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
      obj.assignLabelCoordsIRaw(pos,iPt);

      set(obj.hPts(iPt),'Color',obj.ptsPlotInfo.Colors(iPt,:));
      obj.labeler.labelPosSetI(pos,iPt);
      obj.tfClicked = true;      
      obj.clickedIncrementFrame();
    end    
    
    function kpf(obj,src,evt) %#ok<INUSL>
      key = evt.Key;
      %modifier = evt.Modifier;
      %tfCtrl = any(strcmp('control',modifier));
      
      switch key
        case {'space' 'equal' 'rightarrow' 'd'}
          obj.labeler.frameUpDF(obj.nFrameSkip);
        case {'hyphen' 'leftarrow' 'a'}
          obj.labeler.frameDownDF(obj.nFrameSkip);
      end      
    end
    
    function axOccBDF(obj,src,evt) %#ok<INUSD>
      iPt = obj.iPoint;
      obj.tfOcc(iPt) = true;
      set(obj.hPtsOcc(iPt),'Color',obj.ptsPlotInfo.Colors(iPt,:));
      obj.refreshOccludedPts();
      obj.labeler.labelPosSetOccludedI(iPt);
      tfOcc = obj.labeler.labelPosIsOccluded();
      assert(isequal(tfOcc,obj.tfOcc));
            
      obj.tfClicked = true;
      obj.clickedIncrementFrame();
    end

    function h = getKeyboardShortcutsHelp(obj) %#ok<MANU>
      h = { ...
        '* A/D, LEFT/RIGHT, or MINUS(-)/EQUAL(=) decrements/increments the frame shown.'};
    end
    
    function setIPoint(obj,iPt)
      % set currently labeled point
      
      if ~any(iPt==(1:obj.nPts))
        error('LabelCoreHT:setIPoint','Invalid value for labeling point iPoint.');
      end
      
      obj.iPoint = iPt;
      lObj = obj.labeler;
      lObj.currImHud.updateLblPoint(iPt,obj.labeler.nLabelPoints);
      obj.newFrame([],lObj.currFrame,lObj.currTarget);      
    end    
    
  end
  
  methods (Access=private)       
    
    function clickedIncrementFrame(obj)
      nf = obj.labeler.nframes;
      f = obj.labeler.currFrame;
      df = obj.nFrameSkip;
      iPt = obj.iPoint;
      nPt = obj.nPts;
      if f+df > nf
        if iPt==nPt
          str = sprintf('End of movie reached. Labeling complete for all %d points!',nPt);
          msgbox(str,'Labeling Complete');
        else
          iPt = iPt+1;
          str = sprintf('End of movie reached. Proceeding to labeling for point %d out of %d.',...
            iPt,nPt);
          msgbox(str,'End of movie reached');
          obj.setIPoint(iPt);
          obj.labeler.setFrame(1);
        end
      else
        obj.labeler.frameUpDF(df);
      end
    end
           
  end
  
end