classdef LabelCoreHT < LabelCore
% High-throughput labeling
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
    supportsSingleView = true;
    supportsMultiView = false;
    supportsCalibration = false;
    supportsMultiAnimal = false;
  end

  properties
    iPoint;    % scalar. Either nan, or index of pt currently being labeled
    
    nFrameSkip;
    unlabeledPointColor = [1 1 1];
    otherLabeledPointColor = [0.4 0.4 0.4];
  end

  properties
    unsupportedKPFFns = {} ;  % cell array of field names for objects that have general keypressfcn 
                              % callbacks but are not supported for this LabelCore
  end
  
  methods
    function set.nFrameSkip(obj,val)
      validateattributes(val,{'numeric'},{'positive' 'integer'});
      obj.nFrameSkip = val;
    end
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

      ppi = obj.ptsPlotInfo;
      htm = ppi.HighThroughputMode;
      obj.nFrameSkip = htm.NFrameSkip;
      obj.unlabeledPointColor = htm.UnlabeledPointColor;
      obj.otherLabeledPointColor = htm.OtherLabeledPointColor;
      
      set(obj.hPts,'HitTest','off');
      set(obj.hPtsTxt,'PickableParts','none');
      set(obj.labeler.gdata.tbAccept,'Enable','off');
      obj.labeler.currImHud.updateReadoutFields('hasLblPt',true);

      obj.setIPoint(1);
    end
    
  end
  
  methods
    
    function newFrame(obj,~,iFrm1,iTgt,tfForceUpdate)
      if nargin < 5
        tfForceUpdate = false;
      end
      s = obj.labeler.labelsCurrMovie;
      [tf,p] = Labels.isLabeledFT(s,iFrm1,iTgt);
      xy = reshape(p,[],2);
      %xy = lpos(:,:,iFrm1,iTgt);
      tfUnlbled = isnan(xy(:,1));
      tfLbledOrOcc = ~tfUnlbled;

      iPt = obj.iPoint;
      hPoints = obj.hPts;
      hPointsOcc = obj.hPtsOcc;
      colors = obj.ptsPlotInfo.Colors;      
      
      % POSITIONING
      % - all labeled pts are positioned per labels (including iPoint)
      % - all pure-occed pts are positioned per usual
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
      if ~isempty(hPointsOcc),
        set(hPointsOcc(tfOL),'Color',clr);
      end

      if tfLbledOrOcc(iPt)
        clr = colors(iPt,:);
      else
        clr = obj.unlabeledPointColor;
      end
      set(hPoints(iPt),'Color',clr);
      if ~isempty(hPointsOcc),
        set(hPointsOcc(iPt),'Color',clr);
      end
      
      % MARKER
      % - all labeled or pure-occluded use regular Marker
      % - all labeled and tag-occluded, use OccludedMarker
      % - unlabeled; don't change marker. In particular, for iPoint, leave
      % marker as-is; if last frame was tag-occluded, leave
      % tag-occluded marker
      lpostag = obj.labeler.labeledpostagCurrMovie;
      tfOccTag = lpostag(:,iFrm1,iTgt);
      % tfLbledOrOcc defined above
      
      mrkr = obj.ptsPlotInfo.MarkerProps.Marker;
      mrkrOcc = obj.ptsPlotInfo.OccludedMarker;      
      
      set(hPoints(tfLbledOrOcc & ~tfOccTag),'Marker',mrkr);
      set(hPoints(tfLbledOrOcc & tfOccTag),'Marker',mrkrOcc);      
    end
    
    function newTarget(obj,iTgt0,iTgt1,iFrm) %#ok<INUSD>
      % none; currently multi-target not expected/supported
      % for HT mode
    end
    
    function newFrameAndTarget(obj,~,iFrm1,~,iTgt1,tfForceUpdate)
      if nargin < 6
        tfForceUpdate = false;
      end
      obj.newFrame([],iFrm1,iTgt1,tfForceUpdate);
    end
    
    function clearLabels(obj)
      iPt = obj.iPoint;
      clr = obj.unlabeledPointColor;
      mrkr = obj.ptsPlotInfo.MarkerProps.Marker;
      set(obj.hPts(iPt),'Color',clr,'Marker',mrkr);
      if ~isempty(obj.hPtsOcc),
        set(obj.hPtsOcc(iPt),'Color',clr); % marker should always be mrkr
      end
      obj.labeler.labelPosClearI(iPt);
    end
    
    function acceptLabels(obj) %#ok<MANU>
      assert(false);
    end
    
    function unAcceptLabels(obj) %#ok<MANU>
      assert(false);
    end 
    
    function axBDF(obj,~,evt)
      
      if ~obj.labeler.isReady || evt.Button>1
        return;
      end      
      if obj.isPanZoom(),
        return;
      end


      mod = obj.hFig.CurrentModifier;
      tfShift = any(strcmp(mod,'shift'));
      
      pos = get(obj.hAx,'CurrentPoint');
      pos = pos(1,1:2);
      iPt = obj.iPoint;
      obj.assignLabelCoordsIRaw(pos,iPt);
      
      if ~tfShift
        set(obj.hPts(iPt),...
          'Color',obj.ptsPlotInfo.Colors(iPt,:),...
          'Marker',obj.ptsPlotInfo.MarkerProps.Marker);
        obj.labeler.labelPosTagClearI(iPt);
      else
        set(obj.hPts(iPt),...
          'Color',obj.ptsPlotInfo.Colors(iPt,:),...
          'Marker',obj.ptsPlotInfo.OccludedMarker);
        obj.labeler.labelPosTagSetI(iPt);
      end
      
      obj.labeler.labelPosSetI(pos,iPt);
      obj.clickedIncrementFrame();      
    end
    
    function ptBDF(obj,src,evt) 
      if ~obj.labeler.isReady || evt.Button>1
        return;
      end
      if obj.isPanZoom(),
        return;
      end

      ud = src.UserData;
      if ud==obj.iPoint
        obj.acceptCurrentPt();
      end
    end    
    
    function tfKPused = kpf(obj,src,evt) %#ok<INUSL>
      
      if ~obj.labeler.isReady,
        return;
      end

      
      key = evt.Key;
      modifier = evt.Modifier;
      tfCtrl = any(strcmp('control',modifier));

      tfKPused = true;
      if strcmp(key,'space')
        obj.acceptCurrentPt();
      elseif any(strcmp(key,{'equal' 'rightarrow' 'd'})) && ~tfCtrl
        obj.controller.frameUpDFGUI(obj.nFrameSkip);
      elseif any(strcmp(key,{'hyphen' 'leftarrow' 'a'})) && ~tfCtrl
        obj.controller.frameDownDFGUI(obj.nFrameSkip);
      else
        tfKPused = false;
      end      
    end
    
    function axOccBDF(obj,src,evt) %#ok<INUSD>
      if ~obj.labeler.isReady,
        return;
      end
      if obj.isPanZoom(),
        return;
      end

      iPt = obj.iPoint;
      obj.tfOcc(iPt) = true;
      set(obj.hPtsOcc(iPt),'Color',obj.ptsPlotInfo.Colors(iPt,:));
      obj.refreshOccludedPts();
      obj.labeler.labelPosSetOccludedI(iPt);
      tfOcc = obj.labeler.labelPosIsOccluded();
      assert(isequal(tfOcc,obj.tfOcc));
      
      obj.labeler.labelPosTagClearI(iPt); 
            
      obj.clickedIncrementFrame();
    end

    function h = getLabelingHelp(obj) %#ok<MANU>
      h = { ...
        '* Left-click labels a point and auto-advances the movie.'; ...
        '* Right-click labels an estimate/occluded point and auto-advances the movie.'; ...
        '* Right-click the current point for additional labeling options.'; ...
        '* A/D, LEFT/RIGHT, or MINUS(-)/EQUAL(=) decrements/increments the frame shown.'; ...
        '* <space> accepts the point as-is for the current frame.'};
    end
    
    function setIPoint(obj,iPt)
      % set currently labeled point
      
      if ~any(iPt==(1:obj.nPts))
        error('LabelCoreHT:setIPoint','Invalid value for labeling point iPoint.');
      end
      
      set(obj.hPts,'HitTest','off');
      % clear old contextmenu
      iPtCurr = obj.iPoint;
      if ~isempty(iPtCurr) && ~isnan(iPtCurr)
        obj.hPts(iPtCurr).UIContextMenu = [];
      end
      
      obj.iPoint = iPt;
      
      obj.setupIPointContextMenu();
      set(obj.hPts(iPt),'HitTest','on');
      
      lObj = obj.labeler;
      lObj.currImHud.updateLblPoint(iPt,obj.labeler.nLabelPoints);
      if lObj.currMovie>0
        obj.newFrame([],lObj.currFrame,lObj.currTarget);
      end
    end    
    
  end
  
  methods
    
    function tfEOM = acceptCurrentPt(obj)
      iPt = obj.iPoint;
      hPt = obj.hPts(iPt);
      pos = [hPt.XData hPt.YData];
      
      set(hPt,'Color',obj.ptsPlotInfo.Colors(iPt,:));
      lObj = obj.labeler;
      lObj.labelPosSetI(pos,iPt);
      
      mrkr = hPt.Marker;
      assert(~strcmp(obj.ptsPlotInfo.MarkerProps.Marker,obj.ptsPlotInfo.OccludedMarker),...
        'Marker and OccludedMarker are identical. Please specify distinguishable markers.');
      switch mrkr
        case obj.ptsPlotInfo.MarkerProps.Marker
          lObj.labelPosTagClearI(iPt);
        case obj.ptsPlotInfo.OccludedMarker
          lObj.labelPosTagSetI(iPt);
        otherwise
          assert(false);          
      end
      tfEOM = obj.clickedIncrementFrame();  
    end
   
    function acceptCurrentPtN(obj,nrpt)
      % Equivalent to calling acceptCurrentPt() nrpt times as far as 
      % the actual labeling is concerned; the UI differs, eg the movie does
      % not scroll through all labeled frames
      
      assert(nrpt>0);
      
      iPt = obj.iPoint;
      hPt = obj.hPts(iPt);
      pos = [hPt.XData hPt.YData];      
      
      set(hPt,'Color',obj.ptsPlotInfo.Colors(iPt,:));
      
      lObj = obj.labeler;
      frm0 = lObj.currFrame;
      frmsMax = min(lObj.nframes,frm0+(nrpt-1)*obj.nFrameSkip);
      frms = frm0:obj.nFrameSkip:frmsMax;
      % Note: actual number of repeats may now differ from nrpt
      nrptActual = numel(frms);
      if nrptActual~=nrpt
        str = sprintf('End of movie reached; %d points labeled (over duration of %d frames)',...
          nrptActual,(nrptActual-1)*obj.nFrameSkip);
        msgbox(str,'End of movie');
      end
      
      lObj.labelPosSetFramesI(frms,pos,iPt);
      
      mrkr = hPt.Marker;
      assert(~strcmp(obj.ptsPlotInfo.MarkerProps.Marker,obj.ptsPlotInfo.OccludedMarker),...
        'Marker and OccludedMarker are identical. Please specify distinguishable markers.');
      switch mrkr
        case obj.ptsPlotInfo.MarkerProps.Marker
          lObj.labelPosTagClearFramesI(iPt,frms);
        case obj.ptsPlotInfo.OccludedMarker
          lObj.labelPosTagSetFramesI(iPt,frms);
        otherwise
          assert(false);
      end

      dfrm = frms(end)-frms(1);
      tfEOM = obj.clickedIncrementFrame(dfrm);
      if tfEOM
        warningNoTrace('LabelCoreHT:EOM','End of movie reached.');
      end
    end
    
    function acceptCurrentPtNPrompt(obj)
      resp = inputdlg('Number of times to accept point:','Label current point repeatedly',1,{'1'});
      if isempty(resp)
        % cancel; no-op
        return;
      end
      nrpt = str2double(resp{1});
      if isnan(nrpt) || round(nrpt)~=nrpt || nrpt<=0
        error('LabelCoreHT:input','Input must be a positive integer.');
      end
      obj.acceptCurrentPtN(nrpt);
    end
    
    function acceptCurrentPtNFramesPrompt(obj)
      resp = inputdlg('Accept point over next N frames:','Label current point repeatedly',1,{'1'});
      if isempty(resp)
        % cancel; no-op
        return;
      end
      nfrm = str2double(resp{1});
      if isnan(nfrm) || round(nfrm)~=nfrm || nfrm<=0
        error('LabelCoreHT:input','Input must be a positive integer.');
      end
      nrpt = ceil(nfrm/obj.nFrameSkip);  
      obj.acceptCurrentPtN(nrpt);
    end
    
    function acceptCurrentPtEnd(obj)
      nfrm = obj.labeler.nframes-obj.labeler.currFrame+1;
      nrpt = ceil(nfrm/obj.nFrameSkip);  
      obj.acceptCurrentPtN(nrpt);
    end
    
  end
  
  methods (Access=private)       
    
    function tfEOM = clickedIncrementFrame(obj,dfrm)
      % dfrm (optional): number of frames to increment by. 
      % 
      % tfEOM: true if end-of-movie reached

      if exist('dfrm','var')==0
        dfrm = obj.nFrameSkip;
      end
      
      nf = obj.labeler.nframes;
      f = obj.labeler.currFrame;
      iPt = obj.iPoint;
      nPt = obj.nPts;
      tfEOM = (f+dfrm > nf);
      if tfEOM
        if iPt==nPt
          str = sprintf('End of movie reached. Labeling complete for all %d points!',nPt);
          msgbox(str,'Labeling Complete');
        else
          iPt = iPt+1;
          str = sprintf('End of movie reached. Proceeding to labeling for point %d out of %d.',...
            iPt,nPt);
          msgbox(str,'End of movie reached');
          obj.setIPoint(iPt);
          obj.labeler.setFrameGUI(1);
        end
      else
        obj.controller.frameUpDFGUI(dfrm);
      end
    end
    
    function setupIPointContextMenu(obj)
      c = uicontextmenu(obj.labeler.gdata.mainFigure_);
      hPt = obj.hPts(obj.iPoint);
      hPt.UIContextMenu = c;
      uimenu(c,'Label','Accept point for current frame',...
        'Callback',@(src,evt)obj.acceptCurrentPt);
      uimenu(c,'Label',sprintf('Accept point N times (N*%d frames)',obj.nFrameSkip),...
        'Callback',@(src,evt)obj.acceptCurrentPtNPrompt);
      uimenu(c,'Label',sprintf('Accept point over N frames (N/%d times)',obj.nFrameSkip),...
        'Callback',@(src,evt)obj.acceptCurrentPtNFramesPrompt);
      uimenu(c,'Label','Accept point until end of movie',...
        'Callback',@(src,evt)obj.acceptCurrentPtEnd);      
    end
           
  end
  
end